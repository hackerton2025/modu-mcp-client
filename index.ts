import express from "express";
import cors from "cors";
import { executeCommand } from "./mcp-agent.ts";

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

app.post("/api/execute-command", async (req, res) => {
  try {
    const { command } = req.body;

    if (!command) {
      return res.status(400).json({ error: "명령이 필요합니다" });
    }

    const result = await executeCommand(command);
    res.json({ message: result });
  } catch (error: any) {
    console.error("실행 오류:", error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`✅ 서버가 http://localhost:${PORT}에서 실행 중입니다`);
});
