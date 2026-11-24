# agent_handoff

从 Python 版 **GPRController** 带到 C++ / ROS 版的交接包。内容只有经验、检查清单和工作流，不包含 Python 源码，也不假设 C++ 实现与 Python 相同。

| 文件 | 用途 |
|---|---|
| `CPP_HANDOFF.md` | 主文件：组件映射、已踩 bug 类型、C++/ROS 检查方法、工作流、模板 |
| `AGENTS.snippet.md` | 追加到 C++ 仓库根目录 `AGENTS.md` 的引用片段 |
| `ros-cpp-audit/SKILL.md` | 可选的项目级 skill：可复用的 C++/ROS 审计流程 |

> 默认按 **ROS 2** 写命令；ROS 1 的替代命令已在正文标注。语言为中文，命令/术语保留英文。

---

## 方式 A：复制到 C++ 仓库（推荐）

PowerShell：

```powershell
$cpp = "C:\path\to\your-cpp-repo"

New-Item -ItemType Directory -Force "$cpp\docs\agent" | Out-Null
Copy-Item ".\agent_handoff\CPP_HANDOFF.md" "$cpp\docs\agent\FROM_PYTHON.md"

# 如果 AGENTS.md 不存在，先创建
if (-not (Test-Path "$cpp\AGENTS.md")) { New-Item -ItemType File "$cpp\AGENTS.md" | Out-Null }
Add-Content "$cpp\AGENTS.md" ""
Get-Content ".\agent_handoff\AGENTS.snippet.md" | Add-Content "$cpp\AGENTS.md"

# 可选：项目级 skill（模型在“C++/ROS 找 bug/优化”任务匹配时会加载）
New-Item -ItemType Directory -Force "$cpp\.agents\skills" | Out-Null
Copy-Item -Recurse ".\agent_handoff\ros-cpp-audit" "$cpp\.agents\skills\ros-cpp-audit"
```

Bash：

```bash
CPP_REPO=/path/to/your-cpp-repo
mkdir -p "$CPP_REPO/docs/agent"
cp agent_handoff/CPP_HANDOFF.md "$CPP_REPO/docs/agent/FROM_PYTHON.md"
printf '\n' >> "$CPP_REPO/AGENTS.md"
cat agent_handoff/AGENTS.snippet.md >> "$CPP_REPO/AGENTS.md"

mkdir -p "$CPP_REPO/.agents/skills"
cp -r agent_handoff/ros-cpp-audit "$CPP_REPO/.agents/skills/ros-cpp-audit"
```

复制完成后，在 C++ 仓库新开会话即可：根目录 `AGENTS.md` 会自动加载，并指示 agent 先读 `docs/agent/FROM_PYTHON.md`。

> 如果 C++ 仓库已有 `AGENTS.md`，请先检查内容，避免重复/冲突；片段可以追加，但重复执行会产生重复段落，建议只追加一次或手动合并。
> 如果 `.agents/skills` 与现有工具冲突，也可以把 `ros-cpp-audit` 放到 DSH 的用户级 skills 目录（见 DSH 文档），或只用文档、不装 skill。

---

## 方式 B：不复制，首次会话直接引用绝对路径

在 C++ 仓库的 `AGENTS.md` 里写：

```markdown
## Cross-project prior experience
Read `C:\Users\pc\Desktop\Personal\Code\GPRController\agent_handoff\CPP_HANDOFF.md`
before starting. It contains cross-project lessons from the Python version.
Then copy it into this repository as `docs/agent/FROM_PYTHON.md` and update
this `AGENTS.md` so future sessions load it from the repository.
```

或者第一次对话直接说：

```text
Read <绝对路径>\agent_handoff\CPP_HANDOFF.md first. Then copy it into
docs/agent/FROM_PYTHON.md and add the reference to AGENTS.md.
```

限制：

- 仅限同一台机器；换机器/路径变化会失效；
- 跨 workspace 读取依赖沙箱读权限；
- 长期建议复制进 C++ 仓库（方式 A），让经验随代码版本走。

---

## 在 C++ 会话中的第一条消息（模板）

```text
Follow AGENTS.md. Read docs/agent/FROM_PYTHON.md, then this repo's
docs/agent/ARCHITECTURE.md / RUNBOOK.md / HANDOFF.md.

Start with:
1) clean build + tests + one bag replay (record the baseline);
2) summarize the architecture and current open findings;
3) pick the highest-risk item, reproduce it, and propose a minimal fix + test.

Do not run real-vehicle launch files or change safety limits without explicit approval.
```

如果 C++ 仓库还没有它自己的 `ARCHITECTURE/RUNBOOK/HANDOFF`，第一条任务就可以是“生成这些文档”，但先以 `FROM_PYTHON.md` 为检查清单，不要凭空编造架构。

---

## 更新与维护

- `CPP_HANDOFF.md` 是跨项目经验，建议随 Python 版本的重要教训更新；
- C++ 项目自己的发现、基线、状态应写在 C++ 仓库的 `FINDINGS.md` / `BASELINE.md` / `HANDOFF.md`，不要写回本文件；
- 每次修复后补一条回归测试，并在 C++ 仓库记录“哪条经验被验证/推翻”；
- 不要把实车 IP、密码、传感器序列号等敏感信息写进这些文档。

---

## 安全边界

- 文档默认把实车 launch 视为只读；
- 任何可能让车辆运动的命令都需要显式确认；
- 仿真/rosbag 先复现，实车再验证；
- 修改控制周期、单位、坐标系、QoS、安全限幅属于高风险变更，必须单独评审。
