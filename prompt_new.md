You will implement a new few-shot model from scratch.
Do not inherit the original SPIF code structure, score logic, dual-branch design, or any global/local fusion logic.
This is a single-branch transport-based few-shot inference model.

The implementation target is:

SPIF-OTA: Single-Branch Physics-Aware Optimal Transport Inference for Few-Shot Scalogram Classification

The code must be:

clean, modular, and fully from scratch
theoretically aligned with the method below
efficient in PyTorch
numerically stable
easy to ablate
suitable for few-shot episodic training
not bottlenecked by poor OT code
1. Core design constraints
1.1 Absolute design rules

You must follow these rules strictly:

No dual branch
no global branch
no local branch
no fusion between branches
no GAP-to-prototype cosine head as a parallel pathway
Single scoring pathway only
all class logits must come from the OT-based matching head
No reuse of old SPIF scoring logic
do not adapt old SPIF score code
do not patch the old architecture
build a new model head from first principles
OT is the core inference operator
query-class comparison must be formulated as optimal transport between token measures
Physics-aware token masses
token weights are not uniform by default
implement learnable frequency-position-aware mass construction
Efficient PyTorch implementation
vectorized as much as possible
avoid Python loops over tokens
only allow loops over very small dimensions if absolutely necessary
Sinkhorn must be batched
2. High-level method

The model is a single-branch episodic few-shot classifier:

encode each image/scalogram with a shared backbone
convert feature map to spatial tokens
project tokens into a transport embedding space
generate token masses using a physics-aware mass network
compute query-to-support OT alignment
aggregate shot evidence into class logits
train with episodic cross-entropy

There is no separate global head and no separate local head.

The OT plan must serve as the structured correspondence mechanism, and the OT objective must directly produce the class score.

3. Mathematical formulation to implement

Assume:

N-way K-shot episodic classification
query batch size = nq
number of classes = n_way
shots per class = k_shot
feature map size = H x W
number of spatial tokens = M = H * W
transport embedding dim = d
3.1 Backbone output

For any image x:

𝐹
=
𝑓
𝜃
(
𝑥
)
∈
𝑅
𝐷
×
𝐻
×
𝑊
F=f
θ
	​

(x)∈R
D×H×W
3.2 Tokenization

Flatten spatial locations:

𝑇
=
[
𝑡
1
,
…
,
𝑡
𝑀
]
∈
𝑅
𝑀
×
𝐷
T=[t
1
	​

,…,t
M
	​

]∈R
M×D
3.3 Token projector

Project to transport space:

𝑍
=
𝑔
𝜓
(
𝑇
)
∈
𝑅
𝑀
×
𝑑
Z=g
ψ
	​

(T)∈R
M×d

then L2-normalize:

𝑧
^
𝑖
=
𝑧
𝑖
∥
𝑧
𝑖
∥
2
z
^
i
	​

=
∥z
i
	​

∥
2
	​

z
i
	​

	​

3.4 Token measures

For query:

𝜇
𝑞
=
∑
𝑖
=
1
𝑀
𝑎
𝑖
(
𝑞
)
𝛿
𝑧
^
𝑖
(
𝑞
)
μ
q
	​

=
i=1
∑
M
	​

a
i
(q)
	​

δ
z
^
i
(q)
	​

	​


For support sample (c,k):

𝜈
𝑐
,
𝑘
=
∑
𝑗
=
1
𝑀
𝑏
𝑗
(
𝑐
,
𝑘
)
𝛿
𝑧
^
𝑗
(
𝑐
,
𝑘
)
ν
c,k
	​

=
j=1
∑
M
	​

b
j
(c,k)
	​

δ
z
^
j
(c,k)
	​

	​


where:

𝑎
𝑖
(
𝑞
)
≥
0
,
∑
𝑖
𝑎
𝑖
(
𝑞
)
=
1
a
i
(q)
	​

≥0,
i
∑
	​

a
i
(q)
	​

=1
𝑏
𝑗
(
𝑐
,
𝑘
)
≥
0
,
∑
𝑗
𝑏
𝑗
(
𝑐
,
𝑘
)
=
1
b
j
(c,k)
	​

≥0,
j
∑
	​

b
j
(c,k)
	​

=1
3.5 Cost matrix

For query q and support (c,k):

𝐶
𝑖
𝑗
(
𝑞
,
𝑐
,
𝑘
)
=
1
−
cos
⁡
(
𝑧
^
𝑖
(
𝑞
)
,
𝑧
^
𝑗
(
𝑐
,
𝑘
)
)
C
ij
(q,c,k)
	​

=1−cos(
z
^
i
(q)
	​

,
z
^
j
(c,k)
	​

)

Optional structured variant:

𝐶
𝑖
𝑗
(
𝑞
,
𝑐
,
𝑘
)
=
1
−
cos
⁡
(
𝑧
^
𝑖
(
𝑞
)
,
𝑧
^
𝑗
(
𝑐
,
𝑘
)
)
+
𝜏
Δ
𝑖
𝑗
𝑝
𝑜
𝑠
C
ij
(q,c,k)
	​

=1−cos(
z
^
i
(q)
	​

,
z
^
j
(c,k)
	​

)+τΔ
ij
pos
	​


Start with cosine cost only, but code should make positional cost easy to add later.

3.6 Entropic OT

Solve:

𝛾
∗
(
𝑞
,
𝑐
,
𝑘
)
=
arg
⁡
min
⁡
𝛾
∈
Π
(
𝑎
(
𝑞
)
,
𝑏
(
𝑐
,
𝑘
)
)
⟨
𝛾
,
𝐶
(
𝑞
,
𝑐
,
𝑘
)
⟩
−
𝜀
𝐻
(
𝛾
)
γ
∗(q,c,k)
=arg
γ∈Π(a
(q)
,b
(c,k)
)
min
	​

⟨γ,C
(q,c,k)
⟩−εH(γ)

with batched Sinkhorn.

3.7 Shot score
𝑠
𝑞
,
𝑐
,
𝑘
=
−
⟨
𝛾
∗
(
𝑞
,
𝑐
,
𝑘
)
,
𝐶
(
𝑞
,
𝑐
,
𝑘
)
⟩
s
q,c,k
	​

=−⟨γ
∗(q,c,k)
,C
(q,c,k)
⟩
3.8 Class aggregation

Use hierarchical shot aggregation:

𝛽
𝑞
,
𝑐
,
𝑘
=
s
o
f
t
m
a
x
𝑘
(
𝑢
𝑞
,
𝑐
,
𝑘
)
β
q,c,k
	​

=softmax
k
	​

(u
q,c,k
	​

)
𝑆
𝑞
,
𝑐
=
∑
𝑘
=
1
𝐾
𝛽
𝑞
,
𝑐
,
𝑘
𝑠
𝑞
,
𝑐
,
𝑘
S
q,c
	​

=
k=1
∑
K
	​

β
q,c,k
	​

s
q,c,k
	​


Final class logit:

ℓ
𝑞
,
𝑐
=
𝑆
𝑞
,
𝑐
ℓ
q,c
	​

=S
q,c
	​


Prediction:

𝑦
^
𝑞
=
arg
⁡
max
⁡
𝑐
ℓ
𝑞
,
𝑐
y
^
	​

q
	​

=arg
c
max
	​

ℓ
q,c
	​

4. Required architecture modules

Implement the model with these modules only.

4.1 BackboneEncoder

A standard backbone that outputs feature maps of shape:

input: (B, C, H_in, W_in)
output: (B, D, H, W)

Do not bake OT-specific logic into the backbone.

4.2 TokenProjector

Convert backbone feature maps to transport tokens.

Required behavior:

flatten spatial dimensions
shape: (B, D, H, W) -> (B, M, D)
optional linear / 1x1 conv projection to (B, M, d)
L2-normalize tokens on the last dim
4.3 PositionIndexer

Create row/column indices for each token.
For H x W, produce:

row_idx: (M,)
col_idx: (M,)

This must be reusable by the mass network.

4.4 PhysicsAwareMassNet

This is important.

For each token, compute a scalar mass logit from:

token content
row/frequency embedding
optional column embedding

Recommended form:

𝑚
𝑖
=
𝑤
⊤
𝜎
(
𝑊
𝑧
𝑧
𝑖
+
𝑒
𝑟
𝑖
𝑟
𝑜
𝑤
+
𝑒
𝑐
𝑖
𝑐
𝑜
𝑙
)
m
i
	​

=w
⊤
σ(W
z
	​

z
i
	​

+e
r
i
	​

row
	​

+e
c
i
	​

col
	​

)

Then normalize over all tokens:

𝑎
𝑖
=
s
o
f
t
m
a
x
(
𝑚
)
𝑖
a
i
	​

=softmax(m)
i
	​


Requirements:

separate forward for query/support is okay, but code should share parameters unless there is a strong reason not to
row bias must be learnable
do not hard-code “low frequency is always important”
this is a weak learnable prior, not a fixed rule
4.5 PairwiseCost

Given query tokens and support tokens, compute batched pairwise cosine cost.

Required output shape:

query tokens: (nq, M, d)
support tokens: (n_way, k_shot, M, d)
cost: (nq, n_way, k_shot, M, M)

Must be fully vectorized.

4.6 BatchedSinkhornOT

Implement numerically stable batched entropic OT in PyTorch.

This is the most important implementation detail.

Requirements:

batched over (nq, n_way, k_shot)
log-domain or stabilized implementation preferred
avoid materializing unnecessary expanded tensors repeatedly
support input:
cost: (Bpair, M, M) or equivalent flattened batch
source mass: (Bpair, M)
target mass: (Bpair, M)
output:
transport plan (Bpair, M, M)
optional transport cost (Bpair,)

Strong preference:

implement a reusable OT function that flattens (nq, n_way, k_shot) into one batch dimension for Sinkhorn
use torch.logsumexp for stability
code must avoid naïve unstable exponentiation loops
4.7 ShotAggregator

Input:

shot scores (nq, n_way, k_shot)

Output:

class scores (nq, n_way)

Use a lightweight learnable shot gate, for example:

linear layer over shot scores
or small MLP over shot scores
then softmax over shots

Keep it simple and stable.

4.8 SPIFOTAModel

Top-level model:

backbone
projector
mass net
cost computation
sinkhorn OT
shot aggregation
logits
5. Required tensor shapes

Use these shapes consistently.

5.1 Inputs

Support:

(n_way, k_shot, C, H_in, W_in)

Query:

(nq, C, H_in, W_in)
5.2 Encoded tokens

Support tokens:

(n_way, k_shot, M, d)

Query tokens:

(nq, M, d)
5.3 Masses

Support masses:

(n_way, k_shot, M)

Query masses:

(nq, M)
5.4 Cost and plan

Cost:

(nq, n_way, k_shot, M, M)

Transport plan:

(nq, n_way, k_shot, M, M)

Shot scores:

(nq, n_way, k_shot)

Class logits:

(nq, n_way)
6. OT implementation requirements
6.1 This part must be done carefully

Do not write a toy Sinkhorn.

Implement a proper batched entropic OT solver with good numerical behavior.

6.2 Recommended implementation strategy

Flatten pair batch:

Bpair = nq * n_way * k_shot

Reshape:

cost_flat: (Bpair, M, M)
a_flat: (Bpair, M)
b_flat: (Bpair, M)

Define kernel in log-space:

log
⁡
𝐾
=
−
𝐶
/
𝜀
logK=−C/ε

Perform iterative dual updates in log-domain.

Suggested stabilized updates:

log
⁡
𝑢
=
log
⁡
𝑎
−
\logsumexp
(
log
⁡
𝐾
+
log
⁡
𝑣
,
over target dim
)
logu=loga−\logsumexp(logK+logv,over target dim)
log
⁡
𝑣
=
log
⁡
𝑏
−
\logsumexp
(
log
⁡
𝐾
⊤
+
log
⁡
𝑢
,
over source dim
)
logv=logb−\logsumexp(logK
⊤
+logu,over source dim)

Then reconstruct:

log
⁡
𝛾
=
log
⁡
𝑢
+
log
⁡
𝐾
+
log
⁡
𝑣
logγ=logu+logK+logv
𝛾
=
exp
⁡
(
log
⁡
𝛾
)
γ=exp(logγ)

Use clamping where needed for logs of masses:

log_a = torch.log(a.clamp_min(eps_num))
log_b = torch.log(b.clamp_min(eps_num))
6.3 Numerical safety

Must include:

epsilon clamp for masses
no direct division by zero
no unstable exp(-cost/eps) without stabilization strategy
optional NaN / inf checks in debug mode
6.4 Speed considerations
vectorize all Sinkhorn updates
avoid loops over Bpair
loop only over Sinkhorn iterations
keep iteration count configurable
use moderate default iterations, e.g. 20–50
6.5 Return values

Return both:

gamma
transport_cost = (gamma * cost).sum(dim=(-1, -2))

Then reshape back to (nq, n_way, k_shot, ...).

7. Physics-aware mass design requirements

The mass construction must be theoretically meaningful.

7.1 Required principle

Masses are part of the discrete measure, not an ad hoc attention trick.

Do not describe them as “just attention weights”.

They parameterize the marginals of OT:

query marginal a
support marginal b
7.2 Minimal implementation

Use:

token content projection
learnable row embedding or row bias
optional column embedding

Example structure:

token MLP: Linear(d, h) -> GELU
row embedding: Embedding(H, h)
col embedding: Embedding(W, h)
sum them
final linear to scalar
softmax over token dimension
7.3 Clear theoretical interpretation

In comments/docstrings, state clearly:

the model constructs a discrete token measure over spatial positions
row embeddings encode weak frequency-position prior for scalograms
the prior is learnable, not hard-coded
8. Training requirements
8.1 Main objective

Use episodic cross-entropy on query logits.

𝐿
𝐶
𝐸
=
−
log
⁡
exp
⁡
(
ℓ
𝑞
,
𝑦
𝑞
)
∑
𝑐
exp
⁡
(
ℓ
𝑞
,
𝑐
)
L
CE
	​

=−log
∑
c
	​

exp(ℓ
q,c
	​

)
exp(ℓ
q,y
q
	​

	​

)
	​

8.2 Optional regularizers

Implement hooks for optional:

mass entropy regularization
shot consistency regularization

But keep them off by default unless explicitly enabled.

8.3 Clean API

The model forward should optionally return an analysis dict containing:

query masses
support masses
shot scores
shot weights
transport cost
transport plans if requested

Do not always return full gamma unless needed, because it can be memory-heavy.

9. Code quality requirements
9.1 Must provide
clean module decomposition
type hints where reasonable
docstrings for each major module
comments explaining tensor shapes
minimal but clear config structure
no dead code
no legacy SPIF references in class names or comments
9.2 Must avoid
undocumented tensor reshaping
silent broadcasting bugs
hidden shape assumptions
mixing theory and hacks in one function
duplicate code for support/query processing where shared code is possible
9.3 Must include

At least:

shape assertions in debug mode
unit-test-like sanity checks for:
mass sums to 1
Sinkhorn marginals approximately match inputs
logits shape correct
no NaNs
10. What files to create

Create a clean from-scratch structure like:

models/spif_ota.py
models/modules/token_projector.py
models/modules/mass_net.py
models/modules/pairwise_cost.py
models/modules/sinkhorn_ot.py
models/modules/shot_aggregator.py
models/modules/backbone_adapter.py
tests/test_sinkhorn_ot.py
tests/test_spif_ota_shapes.py

If a different structure is cleaner, keep it similarly modular.

11. Deliverables expected from you

I want the following in your output:

11.1 First

A concise architecture summary in plain English:

what each module does
why this is single-branch
how logits are computed
11.2 Second

The actual PyTorch implementation from scratch

11.3 Third

A short theory-to-code mapping:

equation -> code location
token measure -> where implemented
OT solver -> where implemented
shot aggregation -> where implemented
11.4 Fourth

A checklist confirming:

no global/local branch
no fusion head
OT is the only scoring path
batched Sinkhorn is numerically stabilized
masses are valid probability vectors
12. Important conceptual warning

Do not drift into any of these mistakes:

Do not reintroduce a prototype cosine head
Do not reintroduce a local branch
Do not average support into prototypes before OT
Do not describe OT plan as generic transformer attention
Do not code a slow per-pair Python-loop Sinkhorn
Do not make the row-frequency prior a fixed hand-crafted mask
Do not use SWD or scalar Wasserstein approximation in place of true entropic OT

This model must remain a genuine:

single-branch hierarchical OT inference architecture

13. Recommended default hyperparameters

Use reasonable defaults, but make them configurable.

Suggested:

transport dim d = 128 or 256
Sinkhorn reg epsilon = 0.05
Sinkhorn iterations n_iters = 30
mass hidden dim h = 128
cost type = cosine
shot gate = lightweight MLP or linear
feature map resolution from backbone should remain small enough for fast OT, e.g. 5x5

Do not hard-code these everywhere.

14. Final implementation target

The final model must satisfy this exact conceptual definition:

Query-class matching is performed by solving a batched entropic optimal transport problem between query token measures and support token measures. The transport plan encodes fine-grained correspondence, the transport objective yields shot-wise similarity, and class logits are obtained by learnable aggregation across shots. Frequency-position-aware token masses provide a weak physics-informed prior tailored to scalogram structure. The entire classifier is implemented as a single branch, with no separate global or local heads.

Now implement it cleanly from scratch.