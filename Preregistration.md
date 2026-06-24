**Pre-registration: Positional encoding & induction circuits** _Written before running; predictions committed, reasoning attached. Confidences noted where the call is non-obvious._

**Notation.** RoPE = rotary positional embedding · LPE = learned positional embedding · NoPE = no positional embedding.

**1. Final L1 stripe mass (sharpness of the induction head)**

Prediction: `1 > RoPE > LPE >> NoPE ≈ 0.1` 
NoPE is near the floor: with no positional signal, any mass on the induction offset is leftover from random initialization, not a real circuit. RoPE edges out LPE because its relative-position structure is built in, where LPE has to learn the same offset from scratch. However, this gap is my least confident call (~55%), as nothing forces LPE's learned offset to be blurrier than RoPE's. Across block size I expect mass to _rise_: the tiling generator repeats a fixed-length unit more times in longer sequences, giving the model more in-context evidence that looking back predicts the next token.

**2. Second-half cross-entropy (does induction actually lower loss)** 
Prediction: `4.61 (uniform) > NoPE > LPE > RoPE` 
Ordering follows circuit quality: weaker induction → higher loss, so NoPE sits near the uniform baseline and RoPE lowest. Across block size, loss should _fall_ for LPE and RoPE because more repeats of the unit appear before the second half begins, meaning there is more to copy from. Conversely, it should stay flat for NoPE, which can't exploit the repetition without position.

**3. Formation time (iterations to a working circuit)** 
Prediction: RoPE forms _fastest_, LPE later, NoPE never. (Read from the stripe-mass trajectory's rise, per seed.) Built-in relative position gives RoPE a head start; LPE pays an extra learning cost; NoPE has no circuit to form.

**4. L0 prev-token head: ablation sensitivity of the _loss_** 
Prediction (loss sensitivity): `RoPE > LPE > NoPE` 
Ablating the L0 prev-token head should raise second-half loss most for RoPE. See §6 for the key subtlety: I expect the _mass_ drop to be comparable across modes while the _loss_ cost diverges. So this ranking is about how much RoPE's behavior depends on L0, not about a larger mechanical drop.

**5. Ablation loss ratio (loss after / loss before)** 
Prediction: `RoPE > LPE > NoPE` 
Same mechanism as §4: RoPE leans hardest on the prev-token head, so removing it costs RoPE proportionally more.

**6. Behavioral induction score: the dissociation** 
Pre-ablation: `RoPE > LPE > NoPE`
Post-ablation: `LPE ⪆ RoPE > NoPE`

Two claims at different confidence. The robust one (~75%): RoPE's behavioral score _drops more_ than LPE's under ablation, because RoPE's copying relies more heavily on the single L0 head while LPE leans partly on other contextual routes. The bold one (~50%): that drop is large enough to _invert_ the ranking, meaning RoPE ends up below LPE, not just closer. The inversion needs the drop to exceed RoPE's initial head start, so it's the riskier sub-prediction and I'm marking it as such. NoPE shows little movement either way, as there is no real circuit to disrupt.

**7. The headline: mass drop vs loss increase** 
Prediction: under L0 ablation, RoPE and LPE lose _comparable_ induction mass, but RoPE's second-half loss rises _substantially more_. This is the dissociation I most want on record. If it holds, the L0 prev-token head is mechanically perturbed about equally across modes, yet RoPE's _behavior_ depends on it far more, resulting in the same intervention carrying a divergent behavioral cost.