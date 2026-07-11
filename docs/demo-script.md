# ClaimScope 60-Second Demo

The submitted recording must remain below 60 seconds and include spoken Chinese narration. The recording uses deterministic offline mode and needs no API key. Re-recording requires Windows, Chrome or Edge, the project environment, and the two packages in `scripts/demo-requirements.txt`; the script starts and stops Streamlit itself. Synthetic fixture papers are shown only as UI demonstration data.

| Time | Screen action | Narration |
|---|---|---|
| 0-7 s | Show the default research direction and run Core Claim Arena. | ClaimScope 不直接生成 idea，而是先把模糊研究方向转成可证伪的核心主张。 |
| 7-16 s | Show selected claim, score, falsification test, and open slots. | 结果显式展示机制、目标、预期效果、缺失比较基线、指标和边界条件。 |
| 16-25 s | Scroll through activity and candidates. | LLM 模式由三个 proposer、两个对抗 critic 和一个 judge 交换结构化公开产物；外部服务失败时安全降级。 |
| 25-34 s | Switch to Full Discovery and run the offline workflow. | 完整任务链继续扩展主张变体、拆解隐含假设，并生成支持、反驳、限制和零结果查询。 |
| 34-44 s | Open Evidence and Assumption Ledger. | 证据被映射回每项假设，unknown 与 contradiction 分离，合成 fixture 也会醒目标记。 |
| 44-52 s | Open Opportunities and Trace. | 负结果与悬而未决的假设会变成有边界、可执行的实验机会，并保留任务轨迹。 |
| 52-59 s | Show benchmark and repository end card. | 仓库提供五个 MCP 工具、九十六项通过测试和二十七案例的 70.09 分基线；它是回归指标，不代表科学有效性。 |

## Submission Notes

- File name: `自然语言处理大作业-姓名-学号.mp4`
- Target duration: 55-59 seconds
- Resolution: 1280 x 720 or higher
- Audio: Microsoft Huihui Chinese TTS or a manually recorded replacement
- Do not display an API key, provider request body, or private research text
