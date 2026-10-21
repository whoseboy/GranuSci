# 证据强度-语气校准表（Modality Calibration）

移植自 nature-skills 的 discussion-argument-language。paper12 审稿意见
"void filling, not convection is overclaimed" 的根治方案：结论措辞必须
站在证据能撑住的那一级。

## 措辞阶梯（从弱到强）

| 级别 | 措辞 | 适用证据 |
|---|---|---|
| 0 | is consistent with / is compatible with | 结果与假设共存，未排除替代解释 |
| 1 | suggests / points to | 多个独立信号指向同一方向 |
| 2 | supports / corroborates | 系统性证据 + 排除了部分替代解释 |
| 3 | demonstrates / establishes | 鉴别性实验排除了主要替代解释 |
| 4 | proves / shows definitively | 数学必然或完备枚举（仿真研究几乎永远够不到） |

## 使用规则

1. **先定位证据断点**：问自己"证据链在哪一步断了？"——断点之前的部分才
   是措辞的立足点；
2. **负结果措辞**："未观测到对流环" ≠ "不存在对流"——时均观测排除不了
   瞬时结构（正确写法：time-averaged convection is not observed）；
3. **机制结论的标准格式**："The field data are **consistent with** void
   filling as the working mechanism, **although** direct percolation evidence
   is not presented"（paper12 二轮修改后的写法——正确的降级）；
4. **每条结论自检**：把措辞降一级，结论还成立吗？降一级后更准确的，
   说明原措辞过度；
5. **limitations 写成主张边界**：不写"本研究有局限"（空话），写"结论
   仅在 X 条件（参数域/几何/时标）内成立，Y 条件下的推广未验证"——
   每条局限对应一条主张的适用范围收缩。

## 过度声明的高发场景（自查清单）

- 机制鉴别：只做了"未观测到 A"，就写"排除 A"
- 普适性：单一几何/参数域的结果写"普适标度"
- 外推：拟合范围内的规律外推到范围外
- 首次声明：没做系统查新就说"首次"
- 因果：相关性证据写因果语言
