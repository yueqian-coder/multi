# ClaimScope 60-Second Manual Demo

The submitted recording must remain below 60 seconds and include the student's spoken Chinese narration. Configure the Provider through environment variables before recording so no API key appears on screen. Run `python scripts/probe_provider.py` first and record only when the chosen model is available; do not present a failed or synthetic run as a successful agent result.

| Time | Screen action | Narration |
|---|---|---|
| 0-5 s | Show ClaimScope and toggle `中文 / EN` once. | 我实现的是 ClaimScope，一个在产生研究 idea 之前审计模糊研究方向的双语科研工作台。 |
| 5-16 s | Enter a direction and click Run Arena. Keep the live trace visible. | 系统真正调用三个 proposer、两个对抗 critic 和一个 judge，并实时显示公开执行进度。 |
| 16-25 s | Point to the final claim, structural score, falsification test, and open slots. | 最终主张明确机制、对象、预期效果和证伪实验；这里的分数只表示结构完整性。 |
| 25-35 s | Show candidate comparison and expand one proposer. | 三个提案角色从不同角度操作化研究方向，展开后可以看到公开输入摘要和结构化候选。 |
| 35-44 s | Expand one critic and show rubric scores, reason codes, and revision. | 两个评审按六项量表打分，并公开原因代码和修改建议，但不展示私有思维链。 |
| 44-52 s | Expand Judge, then switch to Full Discovery. | 裁决者只依据公开候选和评审收敛主张；完整流程随后进入真实学术检索。 |
| 52-58 s | Point to the seven-stage workflow and retrieval consent. | 后续会拆解隐含假设，分别检索支持、反驳、限制和零结果，再生成研究机会点。 |
| 58-60 s | Return to title. | 系统还提供 Markdown 导出和五个 MCP 工具，可接入其他科研工作流。 |

## Submission Notes

- File name: `自然语言处理大作业-刘子谦-23354118.mp4`
- Target duration: 55-59 seconds
- Resolution: 1280 x 720 or higher
- Audio: the student's own clear Chinese narration
- Do not display an API key, provider request body, or private research text
- Do not call the public execution trace "chain-of-thought"
- Do not describe the `70.09` benchmark as scientific accuracy
