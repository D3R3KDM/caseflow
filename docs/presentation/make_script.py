"""Builds docs/presentation/script.md (and script.docx) from the speaker notes in deck/project/slides, so the script and
the slides never disagree. As in Assignment 01: the English to say, what it means in Chinese, and tips.

    python docs/presentation/make_script.py        # from the caseflow folder; script.docx needs python-docx
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
SL = HERE / "deck/project/slides"


def note(sid):
    return re.search(r"<aside>(.*?)</aside>", (SL / f"{sid}.html").read_text(encoding="utf-8"), re.S).group(1).strip()


PARTS = [
    ("开场", [
        ("cover", "封面", "0:15", "打招呼；一句话说清项目：模拟我自己电脑机箱里的风、热和灰，并用神经网络在 0.1 秒内预测这些。"),
        ("intro", "Introduction：背景、我的电脑和我做了什么", "0:40",
         "背景：游戏电脑是一盒子热量，显卡和 CPU 加起来能放出 400 多瓦的热。风扇把热带走，但风扇装在哪决定风往哪走、零件有多热。我的电脑：联力 LANCOOL 207 机箱、RTX 5080 公版、锐龙 7 9700X、利民 PA120 SE 散热器；4 个原装风扇（前面 2 个 140 mm、底部 2 个 120 mm，都进风）+ 我自己的 3 个 120 mm（顶中、顶后、后部，都出风）。CaseFlow 只做模拟：所有东西都是设定值，不读这台电脑的硬件数据。我做了三件事：GPU 风洞模拟器、学会它的 3D 神经网络、把两者都画在 3D 里的网页。"),
    ]),
    ("Part 1 · 问题", [
        ("problem", "Eight fan mounts, 6,561 layouts", "0:30",
         "机箱有 8 个风扇位，每个可以空着、进风或出风：3⁸ = 6561 种装法。模拟一种要 90 秒（RTX 5080），全部要约一周，而且这还只是一种室温、一种负载、一套风扇曲线。问题：AI 能不能让这种搜索变成实时的？左图是模拟器“看到”的机箱：6 mm 的小格子，蓝色进风、橙色出风，红色是散热片（发热区）。"),
    ]),
    ("Part 2 · 网页", [
        ("app", "What you see is the network’s prediction", "0:30",
         "过渡句：进入第二部分，网页。屏幕上显示的都是神经网络的预测。机箱在中间，每块屏用细线连到对应的部件（CPU、显卡、风扇、灰尘）。可以改室温和负载、像华硕 Fan Xpert 一样拖风扇曲线、点任意风扇翻转或关掉：网络 0.1 秒给出新装法的结果。"),
        ("compare", "Side by side: the AI and the simulation", "0:35",
         "预测需要一个裁判。点 SIMULATION：AI 的机箱滑到左边，右边“扫描”出模拟的机箱（有缓存约 4 秒，新装法约 90 秒显卡）。标题写着网络有没有见过这个装法；这里是考试装法，网络没见过：CPU 网络说 67.1 °C，模拟是 68.0 °C。打开 Dust，灰跟着各自的风飘进来、积在落下的地方。"),
    ]),
    ("Part 3 · 它怎么工作", [
        ("system", "System and code", "0:30",
         "过渡句：“它在屏幕背后怎么工作？这是第三部分。”三块：引擎（Python + PyTorch，在显卡上算，一种装法只算一次、存进缓存）；FastAPI 小服务（把模拟当成后台任务跑，提供神经网络）；React + three.js 网页（其余全部，包括求温度）。右边是真实代码，约 9,100 行。"),
        ("simulate", "Step 1：模拟一种装法", "0:40",
         "机箱切成 80 × 40 × 80 个 6 mm 的格子；我写的格子玻尔兹曼（LBM）求解器在显卡上一次算三个风扇转速的气流；再算每瓦 CPU / 显卡热量让空气升温多少；再放 4 万颗灰尘颗粒看落在哪。一共约 90 秒。右图上面是我的装法、下面是 32 种里最差的：最差的装法里热空气更多地流到零件上。"),
        ("solve", "Step 2：任何室温、负载、风扇曲线", "0:35",
         "第二步让它变成实时的。热量和功率成正比，所以只存“每瓦升温多少”（3 个转速各一份）。但风扇跟着温度曲线转，转速又影响温度，所以浏览器要找两者吻合的那个点：这里 65.6 °C，我的真机实测 65.3 °C。约 1 毫秒，和完整重新模拟相差不超过 0.51 °C。"),
        ("network", "Step 3：3D U-Net", "0:55",
         "输入就是机箱本身：每个格子 8 个数（实心/空气、机箱外、风扇格和它吹的风速、CPU 和显卡的发热区）+ 坐标。网络把格子缩小 4 次，到 5 × 3 × 5，最深层每个特征都“看得到”整个机箱：顶上一个风扇会改变前面的气流。再放大回来，跳跃连接把 6 mm 的细节带回来。输出是 3D 风速和每瓦升温，和模拟结果的格式完全一样。1350 万个参数，一次预测 0.1 秒。"),
        ("training", "Step 4：训练", "0:35",
         "训练数据是模拟器自己算的：150 种装法（32 种搜索 + 无风扇 + 117 种随机），约 4 小时显卡时间。留 16 种当考试题，网络训练时从来看不到。损失比较每个格子的风和热，再加上两个散热器进风口的空气（零件温度主要由它决定）。300 轮，35 分钟。右图：考试误差在 60 轮左右就不降了，多给数据比多训练有用。"),
    ]),
    ("Part 4 · 结果", [
        ("accuracy", "Accuracy：没见过的装法差半度", "0:35",
         "过渡句：“第四部分，结果。”16 种考试装法 × 6 个场景：CPU 平均差 0.44 °C、显卡 0.28 °C；对照（每种都猜平均值）是 1.04 和 0.74。最差 2.5 °C，是前面没有风扇的装法（训练数据里很少见），满载、35 °C 时。训练过的装法差 0.01 °C：那是背下来的，不算考试。"),
        ("learning", "More data, and dust", "0:30",
         "两个额外检查。只用 1/4、1/2、全部训练装法重新训练：进风口误差 20% → 16% → 10%，所以多模拟一些装法是提高网络最直接的办法。灰尘：网络本身不预测灰，但把模拟器的颗粒放进网络预测的风里，落下的比例和模拟差不到 2 个百分点。"),
        ("search", "All 6,561 layouts", "0:35",
         "收获：网络 4.5 分钟预测了全部 6561 种（模拟要一周）。它找到的最好装法：前面两个风扇往外抽、顶部最前面往外抽、顶部中间往里吹。我用模拟复核了它挑的装法：都在 0.5 °C 以内。右图：4 个放对位置的风扇就到顶了，再多也没用。"),
        ("verdict", "My PC：保留装法，后部别进风", "0:30",
         "对我的电脑意味着什么：保留，我的装法在 32 种里排第 4，比第 1 名差不到 1 度。可选：把后部风扇挪到顶部最前面，仍然往外吹（凉 0.3–1.4 °C）。千万别把后部风扇改成进风：这 12 种在所有场景都是最后 12 名。轻负载时怎么装都差不多。右边：模型和真机实测只差零点几度。"),
    ]),
    ("Part 5 · 过程与总结", [
        ("ai", "Where AI was used", "0:30",
         "过渡句：“最后第五部分，过程。”AI 用在两处：一是用 Claude Code 开发：我定问题和规则、测真机、决定做哪些检验；Claude Code 写了几乎所有代码、跑模拟和检查、出图。二是在网页的功能里：3D U-Net 就是深度学习部分，用 0.1 秒做模拟器 90 秒才做完的事。"),
        ("end", "Takeaways", "0:25", "总结三点和局限，谢谢，请提问。"),
    ]),
]
TIPS = {
    "intro": "提示：这一页少讲，左下角的路线图不用念。讲“我的电脑”时指右边的表。",
    "problem": "提示：讲“3 的 8 次方”时指右边第一个大数字；讲“一周”时指紫色的 ≈ 7 days。左图不用细讲。",
    "app": "**可选现场演示（30 秒）：** 打开 `http://localhost:5190`（提前双击 `caseflow/start.bat`，等顶栏显示 ONLINE）→ Scenario 点 Stress → 拖 Room 滑块 → 点一个前面的风扇 → Turn off → 回到幻灯片。来不及就跳过。",
    "compare": "**可选：放视频** `docs/presentation/caseflow_full_demo.mp4`（3 分 31 秒），第 7–9 段（约 1:20–2:40）是对比和灰尘；或现场点 SIMULATION：你的装法几秒就出来（训练过，两边一样）。想看真实误差：底部 AI → All 6561 → held-out 16 → 点 `F·· B·▸ T▸·◂ R▸` → 关掉 AI 窗口 → SIMULATION。",
    "system": "提示：左边五个方块从上往下讲；右边代码不用逐行讲，指一下 `lbm.py`（LBM 的碰撞和迁移）和 `surrogate.py`（U-Net）两段即可。",
    "simulate": """提示：四行从上往下讲；最后指右图上下两张温度切片的区别（下面更橙 = 更热）。

**格子玻尔兹曼（LBM）通俗解释（自己理解用）：** 不直接解流体方程，而是让每个格子里有 19 群“空气粒子”（一群不动、6 群往面的方向、12 群往棱的方向）。每一步先在格子里碰撞（往平衡状态靠），再沿 19 个方向流到邻居格子。风速和压强从这些粒子群算出来。规则简单、每个格子独立，所以特别适合在显卡上并行。墙把粒子弹回去；风扇格的风速是固定的；机箱外面一圈是敞开的房间。""",
    "solve": "提示：指左图右半边：蓝线和对角线相交的点就是稳态（65.6 °C），虚线是“风扇不跟着变”的情况。",
    "network": """提示：这一页是重点，可以讲慢一点。讲“缩小 4 次”时沿左边的台阶往下指到最底下的黑色方块；讲“放大回来”时沿右边的台阶往上；讲“跳跃连接”时指紫色虚线。

**U-Net 的通俗解释（自己理解用）：**

| 概念 | 通俗解释 |
|---|---|
| 3D 卷积 | 一个 3 × 3 × 3 的小窗口在整个格子里滑动，每处都用同一组权重算出新的特征：看“这一小块周围是什么” |
| 缩小（max-pool） | 每 2 × 2 × 2 个格子取最大值合成一个，格子数变成 1/8，每个特征能“看到”的范围变大 |
| 最底层 5 × 3 × 5 | 每个特征的视野已经覆盖整个机箱，所以能学到“顶上的风扇影响前面的风” |
| 放大 + 跳跃连接 | 把小格子插值回大格子，再把同一层缩小前的特征拼进来，6 mm 的细节（风扇边缘、通风口、散热片）就回来了 |
| 每层的 features | 每个格子用几个数描述：24、48、96、192、384，越往下格子越少、每个格子的数越多 |
| 输出取 log1p(k / 5 mK/W) | 每瓦升温在散热片旁边很大、角落很小，取对数让两头都学得到 |
| 风扇风量不让网络猜 | 每个风扇吹多少风，用和模拟一样的公式先算好，作为输入；网络学的是这些风在机箱里怎么走、怎么把热带走 |""",
    "training": "提示：指右图右半边：浅色虚线（训练装法）一直降到接近 0；深色实线（考试装法）在 60 轮后变平。两条线的差距就是“背下来”和“真会”的区别。",
    "accuracy": "提示：先指左边紫色条（网络）和灰色条（猜平均值）的长短差别，再讲右边三个大数字。“0.01 °C 不算考试”这句要讲清楚，这是上次你问的问题。",
    "learning": "提示：指左图右半边那条往下走的线（20% → 16% → 10%）。",
    "search": "提示：先指左图里中间的“your layout”圆圈；再指右图那条线在 4 个风扇时到底、8 个风扇时又略升。",
    "verdict": "提示：四条从上往下讲；最后指右边深色框里的模型和实测。",
}
QA = [
    ("Why not just simulate every layout?",
     "One layout takes 90 seconds, so all 6,561 take a week, and the scenario changes the answer. The network does them all in 4.5 minutes, and I re-simulate only its best picks.",
     "一种 90 秒，6561 种要一周，而且换个场景答案就变；网络 4.5 分钟算完，我只模拟它挑出来的几种来复核。"),
    ("Is it accurate on a real PC?",
     "In the same game, the model gives 65.6 and 49.7 degrees, and I measured 65.3 and 49.3 on my PC. But that is one PC and one point, and the model has no rising hot air, so trust the differences between layouts more than the absolute numbers.",
     "同一个游戏：模型 65.6 / 49.7 °C，真机实测 65.3 / 49.3 °C。但只有一台电脑一个点，模型里也没有热空气上浮，所以更该相信装法之间的差别，而不是绝对温度。"),
    ("Why a U-Net, not a plain CNN?",
     "The flow is global: a fan on the top changes the air at the front. The four halvings let the deepest layer see the whole case, and the skip connections keep the 6-millimeter detail. A plain CNN would only see nearby cells.",
     "气流是全局的：顶上的风扇会改变前面的风。缩小 4 次让最底层看到整个机箱，跳跃连接保住 6 mm 的细节；普通 CNN 只能看到附近几个格子。"),
    ("How do you know it is not memorizing?",
     "Sixteen layouts were never used in training. On those it is off by 0.44 degrees; on the training layouts by 0.01. Only the held-out numbers count, and the app labels every layout as trained, held-out or new.",
     "16 种考试装法从没参与训练：那里差 0.44 °C；训练过的装法差 0.01 °C。只看考试装法；网页里每种装法都标着“训练过 / 考试题 / 全新”。"),
    ("Why does the error grow with the load?",
     "The network predicts the temperature rise per watt. The same relative error, times more watts, is more degrees: at full stress in a 35-degree room, the worst case is 2.5 degrees.",
     "网络预测的是每瓦升温；同样的相对误差乘上更多的瓦数，换算成的温度就更大：满载、35 °C 室温时最差差 2.5 °C。"),
    ("What is the lattice Boltzmann method?",
     "Instead of solving the flow equations directly, each cell holds 19 groups of particles that collide and stream to the neighbors; the air speed and pressure come out of them. It is simple and runs very well on a GPU.",
     "不直接解流体方程，而是每个格子里 19 群粒子先碰撞、再流到邻居，风速和压强从中算出。规则简单，很适合显卡并行。"),
    ("Why only three fan speeds?",
     "The heat per watt changes smoothly with the fan speed, roughly with one over the speed, so the browser interpolates between three. A full re-simulation at the solved speeds is within 0.51 degrees.",
     "每瓦升温随转速平滑变化（大致和转速成反比），所以浏览器在三个转速之间插值；在求出的转速下完整重新模拟，差不超过 0.51 °C。"),
    ("Where does the dust animation come from?",
     "From the same particle tracking as the engine: dust is released at the openings, drifts with the air, settles and sticks. The browser runs it live in the airflow the case shows, and a test checks that it lands like the engine’s, within one percentage point.",
     "和引擎同一套颗粒追踪：从开口放灰、跟着风飘、沉降、粘住。浏览器在当前显示的风场里实时跑；有测试检查它和引擎落下的比例差不到 1 个百分点。"),
    ("Why not recommend the network’s best layout?",
     "It means reversing the stock front fans for one or one and a half degrees, and the model has no rising hot air, which would hurt a layout that blows down from the top. My layout is already close to the best.",
     "它要把原装前置风扇反过来装，只换来 1–1.5 °C；模型又没有热空气上浮，从顶上往下吹的装法在真机上会吃亏。我现在的装法已经很接近最好。"),
    ("What did you do yourself, and what did AI do?",
     "I chose the problem and the rules, measured my real PC, and decided the tests. Claude Code wrote nearly all the code and ran the simulations and the checks. The project guide records the whole process.",
     "我定问题和规则、测真机、决定做哪些检验；Claude Code 写了几乎所有代码、跑了模拟和检查。项目讲解文档里记录了整个过程。"),
    ("What would you do next?",
     "Simulate more layouts, since the learning curve says data is the limit; add rising hot air; and support other cases and parts.",
     "多模拟一些装法（学习曲线说数据是瓶颈）；加上热空气上浮；支持其他机箱和配件。"),
]

slides = [s for _, ss in PARTS for s in ss]
HINT = re.compile(r"\s*\[If there is time:[^\]]*\]")
words = sum(len(HINT.sub("", note(s[0])).split()) for s in slides)
out = ["# CaseFlow 课堂展示讲稿（10 分钟以内）", "",
       "- 幻灯片：16 页正式内容 + 2 页备用（Claude Slides，可导出 PDF / PPTX；源文件在 `docs/presentation/deck/`，本文件由 `make_script.py` 从幻灯片备注生成）",
       f"- 讲稿约 {words:,} 个英文单词，按每分钟 130 词约 **{words / 130:.1f} 分钟**；现场演示或放视频只在时间够时做",
       "- 结构：开场（背景 + 我的电脑 + 我做了什么 + 五部分路线图）→ Part 1 问题 → Part 2 网页 → Part 3 它怎么工作（系统 + 四步）→ Part 4 结果 → Part 5 过程与总结",
       "- 每页右上角的 1–5 小方块显示当前在第几部分；每部分开头的第一句是过渡句（下面加粗标出）",
       "- 英文句子都很短，可以照着读。中文只帮你理解意思，不用讲", "",
       "| 页 | 部分 | 内容 | 时间 |", "|---|---|---|---|"]
n = 0
for part, ss in PARTS:
    for sid, title, tm, _ in ss:
        n += 1
        out.append(f"| {n} | {part} | {title} | {tm} |")
out += ["| B1, B2 | 备用 | LBM 求解器、工程细节（只在被问到时翻出来） | – |", "", "---", ""]
bridges = ["Part one is the problem.", "That brings me to part two, the app.", "But a prediction needs a referee.",
           "So how does it work? That is part three.", "Part four: results.", "Now the payoff.", "Part five, the process.", "To sum up."]
n = 0
for part, ss in PARTS:
    out += [f"# {part}", ""]
    for sid, title, tm, zh in ss:
        n += 1
        t = note(sid)
        for b in bridges:
            t = t.replace(b, f"**{b}**")
        out += [f"## {n} · {title}（{tm}）", "", f"> {t}", "", f"中文：{zh}", ""]
        if sid in TIPS:
            out += [TIPS[sid], ""]
out += ["# 备用页（被问到时再翻）", "", "## B1 · Inside the lattice-Boltzmann solver", "", f"> {note('solver')}", "",
        "中文：每个格子 19 群粒子，先碰撞（BGK + Smagorinsky 湍流模型）再流到邻居；墙反弹、风扇格固定风速、机箱外是敞开的房间；每个风扇在机箱压力下的工作点；检查：热量 99.9% 从开口带走、温度收敛到 0.1 °C、34 种极端装法要加 4.5 m/s 的限速器。被问到“模拟怎么算的、可靠吗”时用这一页。", "",
        "## B2 · Engineering", "", f"> {note('engineering')}", "",
        "中文：每种装法只模拟一次、存成 8 位 3D 场（约 6 MB）；浏览器里的稳态求解是 Python 版的逐行移植，测试保证差 < 10⁻⁶；积灰动画移植了引擎的颗粒，测试保证落下比例差 < 1 个百分点；网页只保留最近 8 种装法，修好了一次 4 GB 的标签页崩溃；没开引擎也能用；不读任何硬件数据。被问到“怎么做到实时、怎么保证网页和 Python 一致”时用这一页。", "",
        "---", "", "# 可能被问到的问题（英文短答 + 中文）", ""]
for q, a, z in QA:
    out += [f"**{q}**", "", f"> {a}", "", z, ""]
out += ["---", "", "# 交到 Canvas 的东西", "",
        "1. **演示文稿**：打开幻灯片链接 → 右上角 Share / Export → 下载 **PDF**（或 PPTX）",
        "2. **视频**：完整功能演示 `docs/presentation/caseflow_full_demo.mp4`（3 分 31 秒，1920 × 1080，带字幕，13 段：AI 的 3D 结果、场景、风扇曲线、趋势、图层、改装法、AI 对比模拟（训练过的装法 ±0.0、考试装法 −0.9 °C、满载 35 °C −2.5 °C）、积灰、32 种排名、全部 6561 种、配置、方法；生成：在 `caseflow/web` 里 `node scripts/record_full.mjs`，全部用已缓存的模拟，不跑新模拟）",
        "3. **代码**：GitHub 仓库 `github.com/D3R3KDM/caseflow`（或打包 `caseflow` 文件夹，`data/banks/` 是缓存，不用交）",
        "4. **架构和 AI 过程**：幻灯片第 6–10 页和第 15 页；更详细的在 `docs/GUIDE.md`", "",
        "# 上课前准备", "",
        "- 双击 `caseflow/start.bat`：同时打开引擎（:8600）和网页（:5190），等顶栏显示 ONLINE",
        "- 现场点 SIMULATION 对比时，用已经模拟过的装法（你的装法、16 种考试装法）几秒就出来；新装法要等约 90 秒",
        "- 幻灯片用 Present 模式放映，演讲者备注里有本讲稿的英文", ""]
(HERE / "script.md").write_text("\n".join(out), encoding="utf-8")
print("script.md:", words, "words,", n, "slides")


def write_docx(lines, path):
    """The same content as script.md, as a Word document (python-docx)."""
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    INK, MUTED, VIOLET = RGBColor(0x13, 0x15, 0x17), RGBColor(0x5E, 0x63, 0x6A), RGBColor(0x6E, 0x3C, 0xE6)
    doc = Document()
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Inches(0.9)
        sec.top_margin = sec.bottom_margin = Inches(0.8)

    def fonts(style_or_run, latin="Calibri", cjk="Microsoft YaHei"):
        el = style_or_run.element
        rpr = el.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.append(rf)
        for k in ("w:ascii", "w:hAnsi", "w:cs"):
            rf.set(qn(k), latin)
        rf.set(qn("w:eastAsia"), cjk)

    for name in ("Normal", "Title", "Heading 1", "Heading 2", "List Bullet", "List Number"):
        st = doc.styles[name]
        fonts(st)
        st.font.color.rgb = INK
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.styles["Heading 1"].font.size = Pt(16)
    doc.styles["Heading 2"].font.size = Pt(13)
    doc.styles["Heading 2"].font.color.rgb = VIOLET

    def runs(par, text, size=None, color=None):
        for part in re.split(r"(\*\*.+?\*\*|`[^`]+`)", text):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                r = par.add_run(part[2:-2])
                r.bold = True
            elif part.startswith("`") and part.endswith("`"):
                r = par.add_run(part[1:-1])
                fonts(r, "Consolas", "Microsoft YaHei")
            else:
                r = par.add_run(part)
            if size:
                r.font.size = Pt(size)
            if color is not None:
                r.font.color.rgb = color

    def left_rule(par):
        ppr = par._p.get_or_add_pPr()
        bdr = OxmlElement("w:pBdr")
        left = OxmlElement("w:left")
        for k, v in (("w:val", "single"), ("w:sz", "18"), ("w:space", "8"), ("w:color", "6E3CE6")):
            left.set(qn(k), v)
        bdr.append(left)
        ppr.append(bdr)

    first_h1 = True
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[-|\s]+\|$", lines[i]):
                    block.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            t = doc.add_table(rows=len(block), cols=len(block[0]))
            t.style = "Table Grid"
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            t.autofit = False
            weight = [min(max(max(len(row[c]) for row in block), 4), 40) for c in range(len(block[0]))]
            for row in t.rows:
                for c, cell in enumerate(row.cells):
                    cell.width = Inches(6.6 * weight[c] / sum(weight))
            for r, row in enumerate(block):
                for c, cell in enumerate(row):
                    p = t.cell(r, c).paragraphs[0]
                    runs(p, f"**{cell}**" if r == 0 else cell, size=9.5)
            doc.add_paragraph()
            continue
        i += 1
        if not ln.strip() or ln.strip() == "---":
            continue
        if ln.startswith("# "):
            if first_h1:
                doc.add_paragraph(ln[2:], style="Title")
                first_h1 = False
            else:
                doc.add_heading(ln[2:], level=1)
        elif ln.startswith("## "):
            doc.add_heading(ln[3:], level=2)
        elif ln.startswith("> "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.15)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing = 1.25
            left_rule(p)
            runs(p, ln[2:], size=12.5)
        elif ln.startswith("- "):
            runs(doc.add_paragraph(style="List Bullet"), ln[2:])
        elif re.match(r"^\d+\. ", ln):
            runs(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", ln))
        elif ln.startswith("中文："):
            runs(doc.add_paragraph(), ln, size=10, color=MUTED)
        else:
            runs(doc.add_paragraph(), ln)
    doc.save(path)


try:
    write_docx("\n".join(out).split("\n"), HERE / "script.docx")
    print("script.docx written")
except ImportError:
    print("script.docx skipped (pip install python-docx)")
