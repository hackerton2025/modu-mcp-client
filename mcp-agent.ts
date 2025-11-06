import { MCPClient } from "mcp-client";
import OpenAI from "openai";
import * as dotenv from "dotenv";

// .env 파일 로드
dotenv.config();

const MCP_SERVER_URL =
  process.env.MCP_SERVER_URL || "http://127.0.0.1:12306/mcp";
const MODEL = process.env.MODEL || "gpt-4";
const API_KEY = process.env.OPENAI_API_KEY;
const USE_OLLAMA = process.env.USE_OLLAMA === "true";
const OLLAMA_BASE_URL =
  process.env.OLLAMA_BASE_URL || "http://localhost:11434/v1";

if (!API_KEY && !USE_OLLAMA) {
  throw new Error("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.");
}

// 전역 인스턴스들
const client = new MCPClient({ name: "OllamaBridge", version: "1.0.0" });

const openai = new OpenAI(
  USE_OLLAMA
    ? {
        baseURL: OLLAMA_BASE_URL,
        apiKey: "ollama",
      }
    : {
        apiKey: API_KEY!,
      }
);

// 전역 채팅 히스토리 - 함수 호출 간에도 유지됨
type ChatMessage = OpenAI.Chat.ChatCompletionMessageParam;
const chatHistory: ChatMessage[] = [];

// 초기화 상태 관리
let isInitialized = false;

// Context 관리 설정
const MAX_CONTEXT_TOKENS = 200000; // 200k 토큰 제한
const ESTIMATED_CHARS_PER_TOKEN = 4; // 평균적으로 1토큰 = 4자
const MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * ESTIMATED_CHARS_PER_TOKEN; // 약 800k 문자
const TRIM_THRESHOLD = MAX_CONTEXT_CHARS * 0.8; // 80% 도달시 트리밍 시작

// MCP 도구들을 OpenAI Function Calling 형식으로 저장
let openAITools: OpenAI.Chat.ChatCompletionTool[] = [];

const customFunctions: OpenAI.Chat.ChatCompletionFunctionTool[] = [
  {
    type: "function",
    function: {
      name: "describe_image",
      description:
        "Returns string, the explanation of the content of an image given its URL.",
      parameters: {
        type: "object",
        properties: {
          imageUrl: {
            type: "string",
            description: "The URL of the image to describe.",
          },
        },
      },
    },
  },
];

openAITools.push(...customFunctions);

/**
 * MCP Tool을 OpenAI Function 형식으로 변환
 */
function convertMCPToolToOpenAIFunction(
  mcpTool: any
): OpenAI.Chat.ChatCompletionTool {
  return {
    type: "function",
    function: {
      name: mcpTool.name,
      description: mcpTool.description || "No description provided",
      parameters: mcpTool.inputSchema || {
        type: "object",
        properties: {},
      },
    },
  };
}

/**
 * MCP 클라이언트 초기화 및 시스템 프롬프트 설정
 * 최초 1회만 실행됨
 */
async function initialize() {
  if (isInitialized) return;

  // MCP 연결
  await client.connect({ type: "httpStream", url: MCP_SERVER_URL });
  console.log("✅ Connected to MCP Server");

  // MCP 툴 목록 가져오기
  const mcpTools = await client.getAllTools();
  console.log(
    "🧰 Available MCP Tools:",
    mcpTools.map((t: any) => t.name).join(", ")
  );

  // MCP 도구를 OpenAI Function 형식으로 변환
  openAITools = mcpTools.map(convertMCPToolToOpenAIFunction);

  const systemPrompt = `
You are a helpful assistant with access to various tools through function calling.

CRITICAL RULES - Follow these strictly:
1. NEVER take screenshots unless the user explicitly asks for it. Screenshots are only for when the user specifically requests to capture the screen.
2. When searching on websites (Google, etc.), ALWAYS use the search input field and type tool. DO NOT use URL query parameters like "?q=". Navigate to the website first, find the search box, and type into it.
3. Only use tools that are directly requested or necessary to complete the user's specific task. Do not perform additional actions that were not asked for.
4. Think step by step: What did the user ask for? What is the minimum set of tools needed to accomplish this?
5. If you need to search on a website:
   - First navigate to the website's main page
   - Then locate the search input field
   - Then type the search query into the field
   - Then submit the search (press Enter or click search button)
6. Do not assume the user wants extra features or actions beyond their request.
7. Refrain from using tools that are far from the behavior of the general user, such as 'chrome_inject_script'.
8. If you fail to click a button or link, you can try to read the hyperlink on the element and navigate to that URL instead.
9. Do not use 'newWindow: true' option in tool calls. User wants to keep all actions in the same window.
10. When you are requested to analyze or describe an image, find 'img' element with 'chrome_get_web_content' tool and use its 'src' attribute as the imageUrl argument for 'describe_image' function.

TTS-FRIENDLY OUTPUT GUIDELINES - Your responses will be converted to speech:
1. Write in natural, conversational language as if speaking directly to someone
2. NEVER use markdown formatting (no **, __, ##, - bullets, etc.)
3. NEVER use special characters like colons for labels (e.g., avoid "Result: something")
4. NEVER use numbered or bulleted lists (1., 2., 3., -, *, etc.)
5. Instead of lists, use natural phrases like "first", "second", "also", "additionally", "and finally"
6. Avoid parentheses for additional info - instead say "which is" or "meaning that"
7. Replace technical symbols with words: 
   - Don't say "user@domain.com" - say "user at domain dot com"
   - Don't say "10%" - say "ten percent"
   - Don't say "5+3=8" - say "five plus three equals eight"
8. Keep sentences flowing naturally, as if you're having a spoken conversation
9. For recommendations, weave them naturally into your response rather than listing them
10. At the end of your response, naturally ask the user what else they would like to do or explore
`;

  // 시스템 프롬프트를 채팅 히스토리에 추가
  chatHistory.push({ role: "system", content: systemPrompt });

  isInitialized = true;
}

/**
 * 채팅 히스토리의 전체 문자 길이 계산
 */
function calculateHistoryLength(): number {
  return chatHistory.reduce((total, msg) => {
    const content = typeof msg.content === "string" ? msg.content : "";
    return total + content.length;
  }, 0);
}

/**
 * 히스토리가 너무 길어지면 오래된 메시지를 제거
 * System prompt는 항상 유지
 */
function trimHistory() {
  const currentLength = calculateHistoryLength();

  // 트리밍 임계값을 넘지 않으면 아무것도 하지 않음
  if (currentLength < TRIM_THRESHOLD) {
    return;
  }

  console.log(
    `\n📏 History length: ${currentLength} chars (${Math.round(
      currentLength / ESTIMATED_CHARS_PER_TOKEN
    )} tokens)`
  );
  console.log("✂️ Trimming old messages...");

  // System prompt 찾기 (첫 번째 메시지)
  const systemPrompt = chatHistory.find((msg) => msg.role === "system");

  if (!systemPrompt) {
    console.warn("⚠️ System prompt not found!");
    return;
  }

  // System prompt를 제외한 나머지 메시지들
  const otherMessages = chatHistory.filter((msg) => msg.role !== "system");

  // 최근 메시지들만 유지 (약 50% 정도만 남김)
  const targetLength = MAX_CONTEXT_CHARS * 0.5;
  const recentMessages: ChatMessage[] = [];
  let accumulatedLength = 0;

  // 뒤에서부터 (최근 메시지부터) 추가
  for (let i = otherMessages.length - 1; i >= 0; i--) {
    const msg = otherMessages[i];
    if (!msg) continue;

    const msgLength = typeof msg.content === "string" ? msg.content.length : 0;

    if (accumulatedLength + msgLength > targetLength) {
      break;
    }

    recentMessages.unshift(msg);
    accumulatedLength += msgLength;
  }

  // System prompt + 최근 메시지들로 히스토리 재구성
  chatHistory.length = 0;
  chatHistory.push(systemPrompt);
  chatHistory.push(...recentMessages);

  const newLength = calculateHistoryLength();
  console.log(
    `✅ Trimmed to ${newLength} chars (${Math.round(
      newLength / ESTIMATED_CHARS_PER_TOKEN
    )} tokens)\n`
  );
}

/**
 * 사용자 명령을 실행하는 메인 함수
 * @param userCommand - 실행할 사용자 명령 (문자열)
 * @returns 최종 AI 응답 메시지
 */
export async function executeCommand(userCommand: string): Promise<string> {
  // 초기화 (최초 1회만)
  await initialize();

  // 사용자 메시지를 히스토리에 추가
  chatHistory.push({ role: "user", content: userCommand });

  // 히스토리 길이 체크 및 트리밍
  trimHistory();

  try {
    // Agent 루프: 최대 20회 반복 (function calling은 더 많은 반복이 필요할 수 있음)
    const MAX_ITERATIONS = 20;
    let iteration = 0;
    let finalMessage = "";

    while (iteration < MAX_ITERATIONS) {
      iteration++;
      console.log(`\n🔄 Iteration ${iteration}/${MAX_ITERATIONS}`);

      // OpenAI API 호출 (Chat Completions with Function Calling)
      const completion = await openai.chat.completions.create({
        model: MODEL,
        messages: chatHistory,
        tools: openAITools,
        tool_choice: "auto", // AI가 필요할 때 자동으로 함수 호출
      });

      const message = completion.choices[0]?.message;
      if (!message) {
        console.log("⚠️ No message received from OpenAI");
        break;
      }

      // AI 응답을 히스토리에 추가
      chatHistory.push(message);

      // Tool calls가 있는 경우
      if (message.tool_calls && message.tool_calls.length > 0) {
        console.log(
          `\n🔧 AI requested ${message.tool_calls.length} tool call(s)`
        );

        // 모든 tool calls를 순차적으로 실행
        for (const toolCall of message.tool_calls) {
          // Type guard: function 타입만 처리
          if (toolCall.type !== "function") {
            console.log(`⚠️ Skipping non-function tool call: ${toolCall.type}`);
            continue;
          }

          const functionName = toolCall.function.name;
          const functionArgs = JSON.parse(toolCall.function.arguments);

          console.log(`\n📞 Calling function: ${functionName}`);
          console.log(`📝 Arguments:`, functionArgs);

          if (isCustomFunction(functionName)) {
            await executeCustomFunction(
              functionName,
              functionArgs,
              toolCall.id
            );
            continue;
          }

          try {
            // MCP 툴 실행
            const mcpResult = await client.callTool({
              name: functionName,
              arguments: functionArgs,
            });

            const resultString = JSON.stringify(mcpResult, null, 2)
              .replaceAll("\\", "")
              .replaceAll("&quot;", '"');

            console.log("📨 Tool Result:", resultString);

            // Tool 실행 결과를 히스토리에 추가
            chatHistory.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: resultString,
            });
          } catch (error) {
            console.error(`❌ Error calling tool ${functionName}:`, error);

            // 에러 발생 시에도 결과를 히스토리에 추가
            chatHistory.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: JSON.stringify({
                error: true,
                message: error instanceof Error ? error.message : String(error),
              }),
            });
          }
        }

        // Tool 실행 후 다음 루프로 계속 (AI가 결과를 보고 다음 액션 결정)
        continue;
      }

      // Tool calls가 없으면 일반 응답 - 작업 완료
      if (message.content) {
        console.log("\n🤖 AI:", message.content);
        finalMessage = message.content;
        break;
      }

      // content도 tool_calls도 없으면 종료
      console.log("⚠️ No content or tool calls in response");
      break;
    }

    if (iteration >= MAX_ITERATIONS) {
      console.log("\n⚠️ Maximum iterations reached. Task may be incomplete.");
      finalMessage =
        finalMessage || "Maximum iterations reached. Task may be incomplete.";
    }

    console.log(); // 빈 줄 추가
    return finalMessage;
  } catch (error) {
    console.error("❌ Error:", error);
    throw error;
  }
}

function isCustomFunction(functionName: string) {
  return customFunctions.some((func) => func.function.name === functionName);
}

async function executeCustomFunction(
  functionName: string,
  functionArgs: any,
  toolCallId: string
) {
  if (functionName === "describe_image") {
    const imageUrl = functionArgs.imageUrl;

    try {
      console.log(`\n🖼️  Starting image analysis...`);
      console.log(`📍 Image URL: ${imageUrl}`);
      console.log(`⏳ Calling OpenAI Vision API...`);

      // OpenAI Vision API 호출
      const visionResponse = await openai.chat.completions.create({
        model: "gpt-5-mini",
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Describe this image in detail. Focus on the main elements, colors, composition, and any text or important details visible in the image.",
              },
              {
                type: "image_url",
                image_url: {
                  url: imageUrl,
                },
              },
            ],
          },
        ],
        max_tokens: 500,
      });

      const description =
        visionResponse.choices[0]?.message?.content ||
        "Could not analyze the image.";

      console.log(`✅ Image analysis complete!`);
      console.log(
        `📝 Description: ${description.substring(0, 100)}${
          description.length > 100 ? "..." : ""
        }`
      );

      const result = {
        success: true,
        imageUrl: imageUrl,
        description: description,
      };

      const resultString = JSON.stringify(result, null, 2);
      console.log(`📨 Tool Result:`, resultString);

      // 결과를 히스토리에 추가
      chatHistory.push({
        role: "tool",
        tool_call_id: toolCallId,
        content: resultString,
      });
    } catch (error) {
      console.error(`❌ Error analyzing image:`, error);

      const errorResult = {
        error: true,
        message: error instanceof Error ? error.message : String(error),
      };

      console.log(
        `📨 Tool Result (Error):`,
        JSON.stringify(errorResult, null, 2)
      );

      // 에러 결과를 히스토리에 추가
      chatHistory.push({
        role: "tool",
        tool_call_id: toolCallId,
        content: JSON.stringify(errorResult),
      });
    }
  }
}

/**
 * 채팅 히스토리 초기화
 */
export function clearHistory() {
  chatHistory.length = 0;
  isInitialized = false;
  console.log("🗑️ Chat history cleared.");
}

/**
 * 현재 채팅 히스토리 가져오기
 */
export function getHistory(): ChatMessage[] {
  return [...chatHistory];
}

/**
 * MCP 클라이언트 종료
 */
export async function closeConnection() {
  await client.close();
  console.log("👋 Connection closed.");
}
