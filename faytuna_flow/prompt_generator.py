"""Advanced High-Rank & Multi-Domain Calibration Prompt Engine for Faytuna Flow.

Generates scientifically crafted calibration prompts designed to maximize the
effective spectral rank (r_eff) of latent activations in transformer backbones.

Features:
1. Dual-Superposition ("2-in-1"): Fuses two disjoint scientific/technical domains
   into a single context, forcing attention heads to split and SwiGLU gates to widen
   their activation spectrum (counteracting rank deficiency).
2. Deep Code & Architecture Invariants: Strict formulations covering ResNet skip
   connections, atomic lock-free structures, B-Trees, and Raft consensus.
3. Counterfactual & Axiomatic Stress: Modified physical/geometric axioms that
   disable surface n-gram memorization and engage deep multi-layer projection circuits.
4. Multi-Hop Syllogisms: Deep 4-5 step deductive chains testing residual stream retention.
5. Bilingual Semantic Bridges: Cross-lingual technical reasoning linking Russian and
   English latent coordinate charts.
6. Spectral Rank Diagnostics: Mathematical tools to empirically measure the effective
   rank and spectral entropy of representations induced by the prompts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np


# ---------------------------------------------------------------------------
# Domain Knowledge Banks
# ---------------------------------------------------------------------------

CODE_AND_ALGORITHMS = [
    (
        "In deep residual networks, the forward formulation is x_{l+1} = x_l + F(x_l, W_l). "
        "Derive the backward gradient propagation dE/dx_l = dE/dx_L * (I + sum dF/dx_i) to prove "
        "that the identity skip path creates an unobstructed gradient highway preventing vanishing gradients."
    ),
    (
        "def quicksort_hoare_median_of_three(arr, low, high):\n"
        "    mid = (low + high) // 2\n"
        "    # Median-of-three pivot selection prevents worst-case O(n^2) degeneration on pre-sorted arrays\n"
        "    if arr[mid] < arr[low]: arr[low], arr[mid] = arr[mid], arr[low]\n"
        "    if arr[high] < arr[low]: arr[low], arr[high] = arr[high], arr[low]\n"
        "    if arr[high] < arr[mid]: arr[mid], arr[high] = arr[high], arr[mid]\n"
        "    pivot = arr[mid]\n"
        "    # Maintain Hoare loop invariant: all elements in arr[low..i] <= pivot and arr[j..high] >= pivot"
    ),
    (
        "class LockFreeTreiberStack:\n"
        "    # Atomic Compare-And-Swap (CAS) with memory_order_release / memory_order_acquire semantics\n"
        "    # Invariant: Head updates prevent lost updates without mutexes, while Hazard Pointers mitigate the ABA race condition\n"
        "    def __init__(self):\n"
        "        self.head = AtomicReference(None)\n"
        "    def push(self, val):\n"
        "        new_node = Node(val)\n"
        "        while True:\n"
        "            old_head = self.head.get()\n"
        "            new_node.next = old_head\n"
        "            if self.head.compare_and_set(old_head, new_node):\n"
        "                return True"
    ),
    (
        "In modern transformer architectures, SwiGLU replaces standard activation functions via "
        "SwiGLU(x) = Swish(x W_gate) * (x W_up). The elementwise gating mechanism dynamically controls "
        "feature sparsity and enables bilinear interaction between the two projected subspaces."
    ),
    (
        "In the Raft consensus algorithm, log matching invariant dictates that if two distinct server logs "
        "contain an entry with the same index and term, they store the exact same command across all preceding entries."
    ),
    (
        "Red-black tree self-balancing invariants require that: (1) every node is red or black, (2) the root is black, "
        "(3) red nodes cannot have red children, and (4) every simple path from root to leaf contains identical black-height."
    ),
    (
        "In GPU CUDA kernel programming, thread divergence occurs when threads within the same 32-thread warp "
        "execute conflicting conditional execution paths, serializing warp execution and reducing instruction throughput."
    ),
    (
        "Cache-oblivious matrix multiplication achieves optimal memory hierarchy transfers without tuning for specific "
        "L1/L2/L3 cache sizes by recursively subdividing matrices along Morton Z-order space-filling curves."
    ),
]

THEORETICAL_PHYSICS = [
    (
        "The Riemann zeta function analytic continuation to the entire complex plane satisfies the functional equation "
        "zeta(s) = 2^s * pi^(s-1) * sin(pi*s / 2) * Gamma(1-s) * zeta(1-s), placing all non-trivial zeros on Re(s) = 1/2."
    ),
    (
        "In Hamiltonian mechanics, the time evolution of a dynamical system is governed by Hamilton's equations "
        "dq/dt = dH/dp and dp/dt = -dH/dq, which preserve the symplectic 2-form omega = sum dq_i wedge dp_i according to Liouville's theorem."
    ),
    (
        "The second law of thermodynamics establishes that the thermodynamic entropy of an isolated macroscopic system "
        "satisfies dS >= dQ / T, leading to monotonic growth of von Neumann entropy S(rho) = -Tr(rho ln rho) in irreversible processes."
    ),
    (
        "Einstein's field equations G_munu + Lambda g_munu = (8 * pi * G / c^4) * T_munu relate the spacetime metric curvature "
        "directly to the local stress-energy-momentum distribution, dictating that free-falling test particles follow geodesics."
    ),
    (
        "Quantum entanglement in bipartite spin systems violates the classical Bell-CHSH inequality |E(a,b) - E(a,b') + E(a',b) + E(a',b')| <= 2, "
        "reaching the Tsirelson quantum bound of 2 * sqrt(2) through non-local state correlations."
    ),
    (
        "The Navier-Stokes equations for incompressible Newtonian fluids, rho * (du/dt + (u . grad)u) = -grad p + mu * div(grad u), "
        "govern the turbulent transfer of kinetic energy from macroscopic eddies down to the Kolmogorov dissipation scale."
    ),
    (
        "The Fourier transform uncertainty principle states that a function f in L^2(R) and its Fourier transform F(k) "
        "cannot both be sharply localized, satisfying the strict spatial-frequency variance bound Delta_x * Delta_k >= 1/2."
    ),
]

DISCRETE_MATH_AND_LOGIC = [
    (
        "In category theory, a natural transformation eta: F -> G between parallel functors consists of a family of morphisms "
        "such that for every object A, eta_A: F(A) -> G(A) makes the naturality square commute for any morphism f: A -> B."
    ),
    (
        "Gödel's first incompleteness theorem proves that any consistent formal system capable of expressing basic Peano arithmetic "
        "contains undecidable mathematical propositions G such that neither G nor not-G is formally provable within the system."
    ),
    (
        "Galois theory establishes a bijection between subfields of a Galois extension E/F and subgroups of the Galois group Gal(E/F), "
        "demonstrating that polynomial equations of degree 5 or higher are not generally solvable by algebraic radicals."
    ),
    (
        "The Curry-Howard correspondence establishes a direct isomorphism between formal logical systems and computational type theories: "
        "propositions correspond to types, logical proofs correspond to terminating computer programs, and proof reduction corresponds to evaluation."
    ),
    (
        "In spectral graph theory, Cheeger's inequality bounds the graph conductance h(G) via the second smallest eigenvalue of the normalized Laplacian: "
        "lambda_2 / 2 <= h(G) <= sqrt(2 * lambda_2), linking combinatorial bottlenecks directly to continuous diffusion rates."
    ),
]

BIOLOGY_AND_MEDICINE = [
    (
        "The CRISPR-Cas9 endonuclease initiates targeted double-strand DNA breaks by scanning genomic loci for a 5'-NGG-3' PAM sequence, "
        "subsequently unwinding the adjacent duplex to form a guide-RNA/DNA R-loop heteroduplex activating the RuvC and HNH nuclease domains."
    ),
    (
        "The generation and propagation of neuronal action potentials in unmyelinated axons is described by the Hodgkin-Huxley conductance equations, "
        "governed by time- and voltage-dependent activation of Na+ channels and delayed-rectifier K+ channels."
    ),
    (
        "Enzyme kinetics under the Michaelis-Menten steady-state approximation yields reaction velocity v = (V_max * [S]) / (K_m + [S]), "
        "where the Michaelis constant K_m reflects the substrate concentration at which the catalytic rate reaches half its maximum."
    ),
    (
        "In allosteric enzyme regulation and multi-subunit receptor binding, the Hill equation fractional saturation "
        "theta = [L]^n / (K_d + [L]^n) demonstrates how positive cooperativity (Hill coefficient n > 1) transforms "
        "graded chemical inputs into ultra-sensitive, switch-like sigmoidal metabolic response curves."
    ),
]


RUSSIAN_SCIENTIFIC_PROSE = [
    (
        "Основная теорема анализа и формула Ньютона — Лейбница утверждают, что определённый интеграл от непрерывной функции "
        "на замкнутом отрезке равен разности значений её первообразной на границах: интеграл от a до b f(x)dx = F(b) - F(a)."
    ),
    (
        "Принцип наименьшего действия Гамильтона — Остроградского постулирует, что истинная траектория механической системы "
        "сообщает функционалу действия S = интеграл от t1 до t2 L(q, q_dot, t)dt стационарное значение: вариация delta S = 0."
    ),
    (
        "В теории функций комплексного переменного теорема Коши о вычетах позволяет вычислять контурные интегралы замкнутого контура "
        "через сумму вычетов аналитической функции во внутренних изолированных особых точках: контурный интеграл f(z)dz = 2*pi*i * sum Res(f, z_k)."
    ),
    (
        "В статистической физике каноническое распределение Гиббса определяет вероятность нахождения системы в квантовом состоянии с энергией E "
        "как P = (1 / Z) * exp(-E / (k_B * T)), где Z представляет собой статистическую сумму по всем доступным микросостояниям."
    ),
    (
        "Колмогоровская сложность дискретного объекта (битовой строки) определяется как длина наименьшей компьютерной программы "
        "на универсальной машине Тьюринга, способной восстановить и сгенерировать данный объект без потери информации."
    ),
]

HUMANITIES_AND_HISTORY = [
    (
        "The Treaty of Westphalia signed in 1648 established the principle of cuius regio, eius religio "
        "and codified modern Westphalian state sovereignty, delineating non-intervention norms in external affairs."
    ),
    (
        "Following the fall of Constantinople in 1453, the exodus of Greek scholars carrying Byzantine manuscripts "
        "to Italian city-states catalyzed the intellectual revival of Platonic dialectics and the Renaissance humanism."
    ),
    (
        "In Spinoza's Ethics, substance monism posits that God and Nature are identical (Deus sive Natura), "
        "deducing human psychological affects geometrically as necessary modes of an infinite, self-caused substance."
    ),
    (
        "The ancient Eurasian Silk Road operated not merely as a conduit for luxury trade goods, but as a decentralized "
        "network transmitting Sogdian mercantile credit systems, papermaking technology, and Buddhist monastic institutions."
    ),
]

GAME_THEORY_AND_ECONOMICS = [
    (
        "In non-cooperative game theory, a Nash equilibrium represents an action profile where no player has a unilateral "
        "incentive to deviate from their chosen strategy given the fixed equilibrium strategies of all opposing players."
    ),
    (
        "The Arrow-Debreu general equilibrium model mathematically proves the existence of market-clearing competitive prices "
        "using Kakutani's fixed-point theorem under convex production sets and strictly monotonic, concave utility functions."
    ),
    (
        "In mechanism design, the Revelation Principle establishes that any social choice function implementable in Bayesian "
        "Nash equilibrium can be implemented via a direct-revelation, truthful incentive-compatible mechanism."
    ),
    (
        "Black-Scholes-Merton option pricing formalizes continuous-time risk-neutral arbitrage replication via Ito's lemma "
        "and parabolic partial differential equations: dV/dt + 0.5 * sigma^2 * S^2 * d2V/dS2 + r * S * dV/dS - r * V = 0."
    ),
]

CASUAL_AND_CONCISE_DIALOG = [
    (
        "User: Привет! Как дела?\n"
        "Assistant: Привет! Всё отлично, работаю в штатном режиме. Чем могу помочь?"
    ),
    (
        "User: How do I exit vim from command mode?\n"
        "Assistant: Type `:wq` and press Enter to save and exit, or `:q!` to quit without saving."
    ),
    (
        "User: Какая планета Солнечной системы самая близкая к Солнцу?\n"
        "Assistant: Меркурий."
    ),
    (
        "User: What is the square root of 144?\n"
        "Assistant: 12."
    ),
    (
        "User: Спасибо огромное за помощь, всё сработало!\n"
        "Assistant: Отлично! Рад был помочь. Если возникнут ещё вопросы — обращайтесь!"
    ),
    (
        "User: Как в терминале Linux посмотреть текущую рабочую директорию?\n"
        "Assistant: Командой `pwd`."
    ),
    (
        "User: Can you name the three primary colors of light in the additive model?\n"
        "Assistant: Red, Green, and Blue (RGB)."
    ),
    (
        "User: Сколько дней в високосном году?\n"
        "Assistant: 366 дней."
    ),
]

PRACTICAL_CODE_AND_BUGS = [
    (
        "Task: Identify the edge case in this function:\n"
        "```python\n"
        "def compute_average(nums: list[float]) -> float:\n"
        "    return sum(nums) / len(nums)\n"
        "```\n"
        "Analysis: If `nums` is empty, `len(nums)` is 0, raising `ZeroDivisionError`. "
        "Fix: Add `if not nums: return 0.0` at the start."
    ),
    (
        "Task: Write a concise Python expression to invert a key-value dictionary.\n"
        "Solution: `{v: k for k, v in original_dict.items()}` assuming unique values."
    ),
    (
        "Task: Fix the off-by-one boundary error in array iteration:\n"
        "```python\n"
        "for i in range(len(items) + 1):\n"
        "    process(items[i])\n"
        "```\n"
        "Correction: Change `range(len(items) + 1)` to `range(len(items))` or use `for item in items:`."
    ),
    (
        "Task: Explain the difference between `==` and `is` in Python in one sentence.\n"
        "Explanation: `==` checks value equality (whether objects contain equivalent data), "
        "while `is` checks identity (whether both variables point to the exact same memory address)."
    ),
    (
        "Task: Formulate a one-line bash pipeline to find all `.log` files larger than 100MB:\n"
        "Command: `find /var/log -type f -name '*.log' -size +100M`."
    ),
]

COMMONSENSE_AND_DAILY_REASONING = [
    (
        "Question: If a glass bottle filled completely to the brim with liquid water is tightly sealed and placed in a freezer at -18°C, what will happen and why?\n"
        "Explanation: Water expands by approximately 9% upon freezing due to the open hexagonal lattice of ice crystals. Because the rigid glass cannot accommodate this volume expansion, internal pressure will fracture the bottle."
    ),
    (
        "Вопрос: Почему горящее растительное масло на сковороде категорически нельзя заливать водой?\n"
        "Объяснение: Плотность воды выше плотности масла, поэтому вода мгновенно опускается на дно раскаленной сковороды и мгновенно закипает, расширяясь в пар в 1700 раз. Взрыв пара разбрызгивает горящее масло, создавая огненный факел."
    ),
    (
        "Question: A store is open strictly between 09:00 and 18:00 on weekdays. Can a customer make an in-person purchase at 14:30 on a Tuesday?\n"
        "Explanation: Yes, Tuesday is a weekday and 14:30 is between 09:00 and 18:00."
    ),
    (
        "Question: Why do car tires have lower grip and risk hydroplaning on wet asphalt during sudden heavy rain?\n"
        "Explanation: When water volume exceeds the drainage capacity of tire tread grooves, a continuous hydrodynamic wedge of water builds under the tire footprint, lifting rubber off the road and eliminating frictional contact."
    ),
]

SELF_VERIFICATION_AND_INDUCTION = [
    (
        "Task: Determine whether 293 is a prime number via systematic trial division.\n"
        "Step 1: Compute upper limit: ceil(sqrt(293)) = 18.\n"
        "Step 2: Enumerate candidate prime factors p <= 18: [2, 3, 5, 7, 11, 13, 17].\n"
        "Step 3: Test divisibility:\n"
        "- 293 is odd -> not divisible by 2.\n"
        "- Sum of digits is 2 + 9 + 3 = 14 (not divisible by 3).\n"
        "- Last digit is 3 -> not divisible by 5.\n"
        "- 293 = 7 * 41 + 6 -> not divisible by 7.\n"
        "- 293 = 11 * 26 + 7 -> not divisible by 11.\n"
        "- 293 = 13 * 22 + 7 -> not divisible by 13.\n"
        "- 293 = 17 * 17 + 4 -> not divisible by 17.\n"
        "Verification: All primes <= sqrt(293) exhausted with non-zero remainders.\n"
        "Conclusion: 293 is prime."
    ),
    (
        "Task: Evaluate arithmetic Reverse Polish Notation expression: [3, 4, '+', 2, '*', 7, '/'].\n"
        "Step 1: Read 3, 4, operation '+'. Compute 3 + 4 = 7. Current stack: [7].\n"
        "Step 2: Read 2, operation '*'. Compute 7 * 2 = 14. Current stack: [14].\n"
        "Step 3: Read 7, operation '/'. Compute 14 / 7 = 2. Current stack: [2].\n"
        "Verification Check: Input stream empty; stack contains exactly one scalar.\n"
        "Result: 2"
    ),
]

ADVERSARIAL_DISTRACTOR_SUPPRESSION = [
    (
        "Context: A PostgreSQL database server deployed in a Zurich datacenter under cold winter weather "
        "running Linux kernel 5.15 with 64GB DDR5 memory receives a query at 14:02:11 UTC. "
        "The query scans an indexed B-tree containing 1,000,000 leaf pages. "
        "If the B-tree index depth is exactly 3 levels, how many page accesses are required to reach a specific target key?\n"
        "Direct Answer: Exactly 3 page accesses (the root node, one intermediate branch node, and one leaf page). "
        "The geographical location of the server, ambient weather, DDR5 speed, and clock timestamp are completely irrelevant distractors."
    ),
    (
        "Условие: Электропоезд отправляется из Москвы в 10:00 утра со средней скоростью 80 км/ч. "
        "Машиниста зовут Алексей, его стаж составляет 14 лет, а за окном идет мелкий дождь при температуре +11°C. "
        "Расстояние по железнодорожным путям до станции назначения составляет ровно 160 км.\n"
        "Вопрос: Сколько времени продлится поездка до станции назначения?\n"
        "Прямой ответ: 160 км / 80 км/ч = ровно 2 часа. "
        "Имя машиниста, его рабочий стаж, день недели и погодные условия не имеют отношения к расчету."
    ),
]

STRICT_CONTRACT_AND_ANTI_WATER = [
    (
        "Task: Extract system telemetry into strict JSON schema {\"service\": str, \"port\": int, \"healthy\": bool} "
        "without conversational preamble or explanation.\n"
        "Input: 'Nginx reverse proxy operational on port 443 with all health checks passing.'\n"
        "Output: {\"service\": \"nginx\", \"port\": 443, \"healthy\": true}"
    ),
    (
        "Task: Return solely the formal asymptotic time complexity for lookup in a balanced AVL tree.\n"
        "Output: O(log n)"
    ),
    (
        "Task: Return strictly the standard HTTP status code for 'Unauthorized' access.\n"
        "Output: 401"
    ),
]





# ---------------------------------------------------------------------------
# High-Tension Cognitive Stress Generation Templates
# ---------------------------------------------------------------------------

DUAL_SUPERPOSITION_TEMPLATES = [
    (
        "Consider the structural design of deep neural network architectures alongside classical physical dynamics:\n"
        "1. In ResNet models, identity residual skip connections allow unimpeded gradient flow via dE/dx_l = dE/dx_L * (I + sum dF/dx_i).\n"
        "2. In Hamiltonian mechanics, Liouville's theorem proves that the symplectic phase space volume omega = sum dq_i wedge dp_i is strictly preserved.\n"
        "Compare how the preservation of norm and non-dissipative flow in Hamiltonian phase space mathematically relates to the prevention of gradient vanishing in residual streams."
    ),
    (
        "Analyze the intersection of concurrent algorithms and cellular biochemistry:\n"
        "1. A lock-free concurrent queue uses an atomic Compare-And-Swap (CAS) loop to prevent race conditions without acquiring operating system mutexes.\n"
        "2. In cellular enzymatic regulation, allosteric feedback inhibition allows end-products to bind regulatory sites, modulating catalytic velocity without competing for active sites.\n"
        "Contrast the retry-loop mechanism of optimistic concurrency with negative feedback homeostasis under steady-state Michaelis-Menten dynamics."
    ),
    (
        "Examine the duality between discrete graph algorithms and continuous thermodynamics:\n"
        "1. Dijkstra's shortest-path algorithm monotonically expands priority queue frontiers to compute minimum-cost paths across weighted graphs.\n"
        "2. The second law of thermodynamics establishes that entropy in an isolated system monotonically increases over time according to dS >= 0.\n"
        "Explain how the irreversibility of settled vertices in Dijkstra's greedy search mirrors the monotonic arrow of time in thermodynamic entropy."
    ),
    (
        "Synthesize concepts from abstract algebra and transformer representations:\n"
        "1. In category theory, an adjunction between functors F -| G establishes a natural isomorphism Hom(F(A), B) ~= Hom(A, G(B)).\n"
        "2. In transformer attention mechanisms, query-key dot products map representations between token sequence spaces and semantic similarity spaces.\n"
        "Investigate how bidirectional linear projections in multi-head attention resemble adjoint functor pairs preserving structural invariants."
    ),
    (
        "Сопоставьте математический формализм уравнений Навье — Стокса с гидродинамикой информации в глубоких нейросетях:\n"
        "1. В несжимаемой жидкости конвективный член (u . grad)u порождает турбулентный каскад энергии от крупных вихрей к диссипативным масштабам.\n"
        "2. В глубоких трансформерах слои SwiGLU и нелинейные проекции рассеивают представления токенов по латентному пространству.\n"
        "Опишите, как нормализация RMSNorm препятствует взрыву дисперсии аналогично вязкой диссипации кинетической энергии."
    ),
    (
        "Bridge distributed systems consensus with quantum measurement theory:\n"
        "1. The Raft consensus protocol achieves replicated state machine consistency through leader election and unanimous majority quorum commit.\n"
        "2. Quantum state collapse under projective von Neumann measurement forces superposed qubits into an eigenstate of the measurement operator.\n"
        "Compare how the distributed commit point in Raft acts as an irreversible epistemic consensus boundary akin to wave function decoherence."
    ),
    (
        "Analyze tree rebalancing algorithms in relation to economic equilibrium:\n"
        "1. Red-black binary search trees execute tree rotations (left-rotate, right-rotate) to maintain logarithmic O(log n) lookup depth invariants.\n"
        "2. In general equilibrium theory, Walrasian price adjustments shift market supply and demand vectors until excess demand vectors vanish.\n"
        "Contrast local restructuring rotations in search trees with iterative tatonnement price adjustments in market clearing mechanisms."
    ),
    (
        "Examine the mathematical connection between Fourier frequency analysis and kernel methods:\n"
        "1. The Fourier uncertainty principle dictates that temporal signal duration Delta_t and frequency bandwidth Delta_omega satisfy Delta_t * Delta_omega >= 1/2.\n"
        "2. In Support Vector Machines and Gaussian Processes, the radial basis function (RBF) kernel width gamma controls the localization of the reproducing kernel Hilbert space.\n"
        "Demonstrate via Bochner's theorem how the spectral density of stationary kernels governs the trade-off between smoothness and expressive capacity."
    ),
    (
        "Contrast the geometric transport of tangent vectors with consistency in concurrent memory architectures:\n"
        "1. On a curved Riemannian manifold, parallel transport of a vector v along a closed loop yields a phase rotation governed by the Riemann curvature tensor R^l_ijk.\n"
        "2. In multi-core processor architectures, Total Store Order (TSO) hardware consistency enforces FIFO write buffer draining to guarantee linearizability.\n"
        "Analyze how path-dependent holonomy in differential geometry mirrors race-condition non-determinism when memory barriers are omitted."
    ),
    (
        "Unify quantum information boundaries with classical communications channel theory:\n"
        "1. The Holevo bound chi(rho) = S(sum p_i rho_i) - sum p_i S(rho_i) limits the maximum classical information extractable from quantum state ensembles.\n"
        "2. Shannon's noisy-channel coding theorem bounds reliable communication throughput by mutual information capacity C = max I(X; Y).\n"
        "Explain how the subadditivity of quantum von Neumann entropy asymptotically constrains classical communication capacity over quantum channels."
    ),
]


COUNTERFACTUAL_TEMPLATES = [
    (
        "Hypothesize an alternative physical universe governed by non-Euclidean hyperbolic geometry where the parallel postulate is negated:\n"
        "Through any point not on a line, there exist infinitely many distinct parallel lines. "
        "Derive from the Gauss-Bonnet theorem why the sum of interior angles in any triangle must be strictly less than pi, "
        "and explain how the area of a hyperbolic polygon is directly proportional to its angular defect."
    ),
    (
        "Consider an operating system kernel architecture where physical memory addresses are addressed using a 128-bit linear single-level store "
        "without virtual memory page tables or Translation Lookaside Buffers (TLB). "
        "Analyze how spatial memory isolation, permission validation, and cache-line invalidation would be enforced strictly through capabilities and hardware tags."
    ),
    (
        "Suppose a computational model where binary floating-point representations do not support subnormal numbers or NaN values, "
        "saturating instead to maximum finite bounds with fixed precision rounding. "
        "Analyze how catastrophic cancellation and numerical gradient instability would affect backpropagation in deep neural networks."
    ),
    (
        "Представьте математическую систему линейной алгебры, в которой умножение матриц является коммутативным: A * B = B * A для всех квадратных матриц. "
        "Докажите, к какому вырождению спектральной теории операторов и геометрических поворотов в трехмерном пространстве SO(3) это приводит."
    ),
    (
        "Hypothesize a computer architecture where memory access latency is uniformly zero clock cycles across the entire address space, "
        "but the arithmetic logic unit (ALU) latency scales exponentially with bit-width: T(w) = O(2^(w/16)). "
        "Deduce how compiler instruction scheduling, register allocation, and loop unrolling strategies must fundamentally invert their standard trade-offs."
    ),
]


MULTI_HOP_SYLLOGISMS = [
    (
        "Evaluate the following multi-hop formal deductive argument:\n"
        "Premise 1: Any irreducible, aperiodic Markov chain on a finite state space admits a unique stationary distribution pi.\n"
        "Premise 2: The total variation distance to stationarity is bounded by the spectral expansion gap gamma = 1 - lambda_2.\n"
        "Premise 3: A graph perturbation decreases edge cut capacities, causing the normalized Laplacian second eigenvalue lambda_2 to approach 1.\n"
        "Premise 4: The mixing time tau(epsilon) is proportional to 1 / gamma * ln(1 / (epsilon * min(pi))).\n"
        "Deduce the exact asymptotic impact of the perturbation on the convergence rate of random walks on this network."
    ),
    (
        "Trace the causal dependency chain in operating systems storage architecture:\n"
        "Step 1: A user process invokes the write() system call on a memory-mapped file descriptor.\n"
        "Step 2: The virtual memory subsystem marks the corresponding page table entry as dirty without immediate synchronous I/O.\n"
        "Step 3: A background kernel flusher thread initiates asynchronous writeback to block storage.\n"
        "Step 4: The filesystem journal records metadata transactions before committing inode block pointers.\n"
        "Explain how write barriers and journaling prevent filesystem corruption during sudden power termination."
    ),
    (
        "Проследите цепочку логических выводов в теории групп и дифференциальной геометрии:\n"
        "Посылка 1: Группа Ли G является гладким дифференцируемым многообразием, снабженным групповыми операциями.\n"
        "Посылка 2: Касательное пространство в единице группы T_e G образует алгебру Ли g с билинейным коммутатором [X, Y].\n"
        "Посылка 3: Экспоненциальное отображение exp: g -> G переводит элементы алгебры в однопараметрические подгруппы.\n"
        "Выведите, почему локальная геометрия и свойства симметрии группы Ли полностью определяются структурой её алгебры Ли."
    ),
]

BILINGUAL_BRIDGE_TEMPLATES = [
    (
        "Теория графов и алгоритмы оптимизации (Graph Theory and Optimization Algorithms):\n"
        "В задаче о максимальном потоке в сети теорема Форда — Фалкерсона (Max-Flow Min-Cut Theorem) гласит, "
        "что максимальная величина потока из источника s в сток t равна минимальной пропускной способности s-t разреза.\n"
        "Translate this formulation into a rigorous algorithmic specification for the Edmonds-Karp breadth-first search implementation, "
        "and prove that its worst-case computational complexity is strictly bounded by O(V * E^2)."
    ),
    (
        "Дифференциальная геометрия и тензорный анализ (Differential Geometry and Tensor Calculus):\n"
        "Связность Леви-Чивиты на гладком римановом многообразии (M, g) является единственной симметричной связностью, "
        "сохраняющей метрический тензор: ковариантная производная grad_k g_ij = 0.\n"
        "Formulate the Christoffel symbols of the second kind Gamma^k_ij in terms of partial derivatives of metric components, "
        "and express the Riemann curvature tensor R^l_ijk."
    ),
    (
        "Машинное обучение и оптимизация (Machine Learning and Optimization Theory):\n"
        "Метод стохастического градиентного спуска с адаптивной оценкой моментов (Adam Optimizer) вычисляет экспоненциально "
        "взвешенные скользящие средние первого момента градиента m_t и второго момента v_t.\n"
        "Provide the exact bias correction equations for m_hat_t and v_hat_t and explain why the bias correction factor "
        "(1 - beta^t) is critical during the initial training iterations when t is small."
    ),
]


# ---------------------------------------------------------------------------
# Calibration Suite Generator Class
# ---------------------------------------------------------------------------

class CalibrationPromptEngine:
    """Production-grade generator for high-rank, diverse calibration prompts."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def get_curated_baseline_prompts(self) -> list[str]:
        """Return curated high-rank baseline prompts across core domains."""
        baseline = []
        baseline.extend(CODE_AND_ALGORITHMS[:4])
        baseline.extend(THEORETICAL_PHYSICS[:4])
        baseline.extend(DISCRETE_MATH_AND_LOGIC[:3])
        baseline.extend(BIOLOGY_AND_MEDICINE[:3])
        baseline.extend(RUSSIAN_SCIENTIFIC_PROSE[:4])
        return baseline

    def get_dual_superposition_prompts(self) -> list[str]:
        """Return cross-domain '2-in-1' superposition stress prompts."""
        return list(DUAL_SUPERPOSITION_TEMPLATES)

    def get_counterfactual_prompts(self) -> list[str]:
        """Return counterfactual reasoning stress prompts."""
        return list(COUNTERFACTUAL_TEMPLATES)

    def get_multi_hop_prompts(self) -> list[str]:
        """Return multi-hop syllogistic reasoning prompts."""
        return list(MULTI_HOP_SYLLOGISMS)

    def get_bilingual_bridge_prompts(self) -> list[str]:
        """Return Russian-English bilingual bridging prompts."""
        return list(BILINGUAL_BRIDGE_TEMPLATES)

    def generate_calibration_suite(
        self,
        num_prompts: int = 64,
        dual_ratio: float = 0.35,
        code_ratio: float = 0.25,
        math_phys_ratio: float = 0.20,
        russian_ratio: float = 0.20,
    ) -> list[dict[str, Any]]:
        """Generate a structured, diverse calibration suite.

        Args:
            num_prompts: Total target prompts to assemble.
            dual_ratio: Fraction of dual-superposition ("2-in-1") prompts.
            code_ratio: Fraction of code and structural invariant prompts.
            math_phys_ratio: Fraction of theoretical physics/math prompts.
            russian_ratio: Fraction of Russian/bilingual scientific prompts.

        Returns:
            List of prompt record dicts with metadata.
        """
        records: list[dict[str, Any]] = []

        # Tier 1: Casual & Concise Dialog (Protects against Encyclopedia Syndrome / register collapse)
        for idx, text in enumerate(CASUAL_AND_CONCISE_DIALOG):
            records.append({
                "id": f"casual_concise_{idx:02d}",
                "category": "casual_concise_dialog",
                "text": text,
                "strategy": "register_natural_brevity",
                "estimated_tension": "low",
            })

        # Tier 2: Practical Code & Bug Spotting (Direct engineering execution without fluff)
        for idx, text in enumerate(PRACTICAL_CODE_AND_BUGS):
            records.append({
                "id": f"practical_code_{idx:02d}",
                "category": "practical_code_and_bugs",
                "text": text,
                "strategy": "targeted_bug_isolation",
                "estimated_tension": "medium",
            })

        # Tier 3: Everyday Commonsense & Cause-and-Effect Reasoning
        for idx, text in enumerate(COMMONSENSE_AND_DAILY_REASONING):
            records.append({
                "id": f"commonsense_{idx:02d}",
                "category": "commonsense_reasoning",
                "text": text,
                "strategy": "physical_cause_and_effect",
                "estimated_tension": "medium",
            })

        # 4. Dual Superposition ("2-in-1") High-Rank Stress
        duals = self.get_dual_superposition_prompts()
        for idx, text in enumerate(duals):
            records.append({
                "id": f"dual_superposition_{idx:02d}",
                "category": "dual_superposition_2in1",
                "text": text,
                "strategy": "cross_domain_superposition",
                "estimated_tension": "high",
            })

        # 5. Code & Structural Invariants
        for idx, text in enumerate(CODE_AND_ALGORITHMS):

            records.append({
                "id": f"code_invariant_{idx:02d}",
                "category": "code_and_algorithms",
                "text": text,
                "strategy": "deep_structural_invariant",
                "estimated_tension": "medium",
            })

        # 3. Theoretical Physics & Math
        for idx, text in enumerate(THEORETICAL_PHYSICS):
            records.append({
                "id": f"physics_math_{idx:02d}",
                "category": "theoretical_physics",
                "text": text,
                "strategy": "continuous_geometry_and_physics",
                "estimated_tension": "high",
            })

        # 4. Discrete Math & Logic
        for idx, text in enumerate(DISCRETE_MATH_AND_LOGIC):
            records.append({
                "id": f"discrete_logic_{idx:02d}",
                "category": "discrete_math_and_logic",
                "text": text,
                "strategy": "formal_reasoning",
                "estimated_tension": "high",
            })

        # 5. Biology & Medicine
        for idx, text in enumerate(BIOLOGY_AND_MEDICINE):
            records.append({
                "id": f"biology_med_{idx:02d}",
                "category": "biology_and_medicine",
                "text": text,
                "strategy": "biochemical_dynamics",
                "estimated_tension": "medium",
            })

        # 6. Counterfactual & Syllogism
        for idx, text in enumerate(COUNTERFACTUAL_TEMPLATES):
            records.append({
                "id": f"counterfactual_{idx:02d}",
                "category": "counterfactual_axiomatic",
                "text": text,
                "strategy": "axiom_inversion",
                "estimated_tension": "maximum",
            })

        for idx, text in enumerate(MULTI_HOP_SYLLOGISMS):
            records.append({
                "id": f"syllogism_{idx:02d}",
                "category": "multi_hop_syllogism",
                "text": text,
                "strategy": "deductive_chain",
                "estimated_tension": "high",
            })

        # 7. Russian & Bilingual Bridge
        for idx, text in enumerate(RUSSIAN_SCIENTIFIC_PROSE):
            records.append({
                "id": f"russian_science_{idx:02d}",
                "category": "russian_scientific_prose",
                "text": text,
                "strategy": "multilingual_monolingual_russian",
                "estimated_tension": "high",
            })

        for idx, text in enumerate(BILINGUAL_BRIDGE_TEMPLATES):
            records.append({
                "id": f"bilingual_bridge_{idx:02d}",
                "category": "bilingual_semantic_bridge",
                "text": text,
                "strategy": "cross_lingual_mapping",
                "estimated_tension": "maximum",
            })

        # 8. Humanities & History
        for idx, text in enumerate(HUMANITIES_AND_HISTORY):
            records.append({
                "id": f"humanities_history_{idx:02d}",
                "category": "humanities_and_history",
                "text": text,
                "strategy": "historical_epistemology",
                "estimated_tension": "medium",
            })

        # 9. Game Theory & Economics
        for idx, text in enumerate(GAME_THEORY_AND_ECONOMICS):
            records.append({
                "id": f"game_theory_econ_{idx:02d}",
                "category": "game_theory_and_economics",
                "text": text,
                "strategy": "equilibrium_and_optimization",
                "estimated_tension": "high",
            })

        # 10. Self-Verification & Induction Chains
        for idx, text in enumerate(SELF_VERIFICATION_AND_INDUCTION):
            records.append({
                "id": f"self_verification_{idx:02d}",
                "category": "self_verification_induction",
                "text": text,
                "strategy": "in_context_sanity_check",
                "estimated_tension": "high",
            })

        # 11. Adversarial Distractor Suppression
        for idx, text in enumerate(ADVERSARIAL_DISTRACTOR_SUPPRESSION):
            records.append({
                "id": f"adversarial_distractor_{idx:02d}",
                "category": "adversarial_distractor_suppression",
                "text": text,
                "strategy": "irrelevant_token_filtering",
                "estimated_tension": "maximum",
            })

        # 12. Strict Contract & Anti-Water Formats
        for idx, text in enumerate(STRICT_CONTRACT_AND_ANTI_WATER):
            records.append({
                "id": f"strict_contract_{idx:02d}",
                "category": "strict_contract_anti_water",
                "text": text,
                "strategy": "zero_fluff_precision",
                "estimated_tension": "high",
            })

        # 13. Combinatorial Dual-Superposition ("2-in-1") Synthesis if more prompts needed
        if len(records) < num_prompts:

            domain_pairs = [
                ("Code & Algorithms", CODE_AND_ALGORITHMS, "Theoretical Physics", THEORETICAL_PHYSICS),
                ("Formal Logic", DISCRETE_MATH_AND_LOGIC, "Biochemical Dynamics", BIOLOGY_AND_MEDICINE),
                ("Russian Science", RUSSIAN_SCIENTIFIC_PROSE, "Code & Invariants", CODE_AND_ALGORITHMS),
                ("Game Theory", GAME_THEORY_AND_ECONOMICS, "Continuous Physics", THEORETICAL_PHYSICS),
                ("History & Sovereignty", HUMANITIES_AND_HISTORY, "Graph Theory & Networks", DISCRETE_MATH_AND_LOGIC),
                ("Quantum Dynamics", THEORETICAL_PHYSICS, "Lock-Free Systems", CODE_AND_ALGORITHMS),
                ("Enzyme Kinetics", BIOLOGY_AND_MEDICINE, "Thermodynamics", THEORETICAL_PHYSICS),
                ("Russian Mathematics", RUSSIAN_SCIENTIFIC_PROSE, "Game Theory & Economics", GAME_THEORY_AND_ECONOMICS),
            ]
            synth_idx = 0
            while len(records) < num_prompts:
                d1_name, d1_list, d2_name, d2_list = domain_pairs[synth_idx % len(domain_pairs)]
                t1 = d1_list[(synth_idx // len(domain_pairs)) % len(d1_list)]
                t2 = d2_list[(synth_idx // len(domain_pairs)) % len(d2_list)]
                synth_text = (
                    f"Cross-Domain Dual-System Analysis & Conceptual Synthesis:\n"
                    f"[System 1: {d1_name}]\n{t1}\n\n"
                    f"[System 2: {d2_name}]\n{t2}\n\n"
                    f"Requirement: Formulate a rigorous mathematical or structural analogy comparing the "
                    f"invariance principle in System 1 with the stability or convergence condition in System 2."
                )
                records.append({
                    "id": f"synth_dual_superposition_{synth_idx:03d}",
                    "category": "dual_superposition_2in1",
                    "text": synth_text,
                    "strategy": "combinatorial_cross_domain_superposition",
                    "estimated_tension": "maximum",
                })
                synth_idx += 1

        # If requested more than total available, slice exactly to num_prompts
        if len(records) > num_prompts:
            records = records[:num_prompts]

        return records


    def save_calibration_dataset(
        self,
        filepath: str | Path,
        records: Sequence[dict[str, Any]],
    ) -> Path:
        """Save calibration prompts to structured JSON."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "2.0.0",
            "total_prompts": len(records),
            "categories": list({r["category"] for r in records}),
            "prompts": list(records),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_path

    @staticmethod
    def load_calibration_dataset(filepath: str | Path) -> list[str]:
        """Load text prompts from structured calibration dataset JSON."""
        in_path = Path(filepath)
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "prompts" in data:
            return [p["text"] if isinstance(p, dict) else str(p) for p in data["prompts"]]
        elif isinstance(data, list):
            return [p["text"] if isinstance(p, dict) else str(p) for p in data]
        raise ValueError(f"Unrecognized calibration JSON schema at {filepath}")


# ---------------------------------------------------------------------------
# Representation Geometry & Spectral Rank Diagnostic Tools
# ---------------------------------------------------------------------------

def compute_spectral_effective_rank(matrix: np.ndarray, eps: float = 1e-12) -> float:
    """Compute the Roy-Vetterli spectral effective rank of a matrix X.

    Effective rank r_eff = exp(H(sigma)) where H is normalized singular entropy:
        p_i = sigma_i / sum(sigma_j)
        H = - sum(p_i * ln(p_i))
        r_eff = exp(H)

    Properties:
        - 1.0 <= r_eff <= min(M, N).
        - Measures the effective number of non-dormant orthogonal dimensions.
    """
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return 1.0
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    s_sum = float(np.sum(s))
    if s_sum < eps:
        return 1.0
    p = s / s_sum
    p = p[p > eps]
    entropy = -float(np.sum(p * np.log(p)))
    eff_rank = float(np.exp(entropy))
    return eff_rank


def compute_spectral_entropy(matrix: np.ndarray, eps: float = 1e-12) -> float:
    """Compute normalized Shannon spectral entropy H in [0, 1]."""
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return 0.0
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    s_sum = float(np.sum(s))
    if s_sum < eps:
        return 0.0
    p = s / s_sum
    p = p[p > eps]
    r = len(s)
    if r <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(r))
