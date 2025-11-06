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
  const tools = await client.getAllTools();
  console.log(
    "🧰 Available MCP Tools:",
    tools.map((t: any) => t.name).join(", ")
  );

  // MCP 도구 안내 문구(system prompt) 생성
  const toolDescriptions = tools
    .map((t: any) => `- ${t.name}: ${t.description || "No description"}`)
    .join("\n");

  const systemPrompt = `
You are an assistant connected to an MCP server.
You can call the following tools by outputting a JSON object in this format:

{
  "tool": "<tool_name>",
  "arguments": { ... }
}

Available tools:
${toolDescriptions}

When you want to use a tool, output only the JSON object (no explanation or extra text).

IMPORTANT: You can use multiple tools in sequence to complete a task.
After each tool execution, you will see the result and can decide to:
1. Use another tool by outputting another JSON object
2. Complete the task by outputting: {"done": true, "message": "your final response to the user"}
3. You can call only one tool at once. Do not output any text after the first JSON object.

When you output {"done": true, "message": "..."}, the conversation will end and the user will see your final message.
In the final message, please ask the user what else they would like to do next with recommendations.

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
8. If you fails to click a button or link, you can try read hyperlink on the element and navigate to that URL instead.

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

GOOD TTS EXAMPLE: 
"I found three interesting options for you. First, there's a Italian restaurant nearby with great reviews. Second, you might like the new sushi place that just opened. And finally, there's a cozy cafe that serves excellent pastries. What would you like to know more about?"

BAD TTS EXAMPLE:
"Here are the results:
1. **Italian Restaurant** - Great reviews (4.5/5)
2. **Sushi Place** - Newly opened
3. **Cafe** - Excellent pastries

What else would you like to do?"
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
    // Agent 루프: 최대 10회 반복
    const MAX_ITERATIONS = 10;
    let iteration = 0;
    let taskComplete = false;
    let finalMessage = "";

    while (!taskComplete && iteration < MAX_ITERATIONS) {
      iteration++;
      console.log(`\n🔄 Iteration ${iteration}/${MAX_ITERATIONS}`);

      // OpenAI API 호출 (Chat Completions)
      const completion = await openai.chat.completions.create({
        model: MODEL,
        messages: chatHistory,
      });

      const llmOutput = completion.choices[0]?.message?.content?.trim() || "";
      console.log("\n🤖 AI:", llmOutput);

      // LLM 응답을 히스토리에 추가
      chatHistory.push({ role: "assistant", content: llmOutput });

      // LLM 출력이 JSON인지 확인
      try {
        const parsed = JSON.parse(llmOutput);

        // 작업 완료 확인
        if (parsed.done === true) {
          console.log("\n✅ Task completed!");
          if (parsed.message) {
            console.log("📝 Final message:", parsed.message);
            finalMessage = parsed.message;
          }
          taskComplete = true;
          break;
        }

        // MCP 툴 실행
        if (parsed.tool && parsed.arguments) {
          console.log(`\n🔧 Using tool: ${parsed.tool}`);
          const mcpResult = await client.callTool({
            name: parsed.tool,
            arguments: parsed.arguments,
          });

          const resultString = JSON.stringify(mcpResult, null, 2);
          console.log("📨 Tool Result:", resultString);

          // MCP 결과를 히스토리에 추가 (시스템 메시지로)
          chatHistory.push({
            role: "system",
            content: `Tool execution result:\nTool: ${parsed.tool}\nResult: ${resultString}`,
          });

          // 매 툴 실행 후 히스토리 체크
          trimHistory();
        } else {
          // JSON이지만 tool이나 done이 없는 경우
          console.log("⚠️ Invalid JSON format. Ending iteration.");
          taskComplete = true;
        }
      } catch {
        // JSON이 아닌 경우는 일반 응답으로 처리하고 종료
        finalMessage = llmOutput;
        taskComplete = true;
      }
    }

    if (iteration >= MAX_ITERATIONS && !taskComplete) {
      console.log("\n⚠️ Maximum iterations reached. Task may be incomplete.");
      finalMessage = "Maximum iterations reached. Task may be incomplete.";
    }

    console.log(); // 빈 줄 추가
    return finalMessage;
  } catch (error) {
    console.error("❌ Error:", error);
    throw error;
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
