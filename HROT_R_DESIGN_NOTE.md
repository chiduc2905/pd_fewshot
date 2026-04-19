# HROT-R: Tu Van De Few-Shot Den Van Tai Toi Uu Co Cau Truc

## Tom Tat

HROT-R la mot bien the cua ho mo hinh HROT-FSL cho bai toan phan loai anh scalogram phong dien cuc bo trong che do few-shot. Tai moi episode, mo hinh chi co vai anh ho tro cho moi lop, nhung van phai phan loai cac anh truy van moi. Kho khan chinh nam o cho dau hieu phong dien thuong chi xuat hien cuc bo tren mien thoi gian-tan so, bi bao quanh boi background noise, va cac support shot cung lop co the khong cung chua dung mot mau bang chung.

Dong HROT giai bai toan nay bang cach xem moi anh nhu mot tap token cuc bo va so khop query voi support bang unbalanced optimal transport. Unbalanced OT cho phep chi van chuyen phan khoi luong bang chung dang tin cay thay vi ep moi vung trong anh query phai khop voi moi vung trong support. HROT-H dat nen mong bang transported mass thich nghi theo episode va diem so co nguong chi phi. HROT-Q bo sung uoc luong do tin cay token, support consensus, noise sink, va robust shot pooling de xu ly background noise.

HROT-R xuat hien tu mot diem yeu con lai cua HROT-Q: Q biet token nao dang tin, nhung van cham diem moi cap token chu yeu theo chi phi dac trung cap mot. Dieu nay co the lam mo hinh tin vao mot vai cap token co chi phi thap tinh co, du cac cap do khong bao toan cau truc cuc bo cua mau phong dien. HROT-R them mot chi phi nhat quan cau truc, duoc suy ra tu posterior UOT plan cua Q, de hoi them mot cau hoi:

```text
Neu query token i duoc ghep voi support token j,
thi quan he cua i voi cac token query khac co giong
quan he cua j voi cac token support tuong ung hay khong?
```

Noi ngan gon, HROT-R khong chi hoi "hai token co giong nhau khong", ma hoi "cap ghep nay co nam trong mot mau quan he cuc bo hop ly khong".

Chay variant R bang:

```bash
python run_all_experiments.py --models hrot_fsl --hrot_variant R
```

Co the dieu chinh do manh ban dau cua chi phi cau truc:

```bash
--hrot_structure_cost_init 0.05
```

## 1. Bai Toan: Few-Shot Phan Loai Scalogram Phong Dien

Trong few-shot classification, mo hinh khong duoc huan luyen lai rieng cho tung lop moi voi nhieu du lieu. Thay vao do, moi episode gom:

- `N` lop can phan loai;
- `K` anh support cho moi lop;
- mot tap anh query can gan nhan.

Neu la `5-way 1-shot`, moi episode co 5 lop va moi lop chi co 1 anh support. Neu la `5-way 5-shot`, moi lop co 5 anh support.

Voi anh scalogram phong dien cuc bo, dau hieu phan biet lop thuong co ba tinh chat:

1. **Cuc bo**: bang chung co the chi nam o mot vung thoi gian-tan so nho.
2. **Khong day du**: hai mau cung lop co the khong hien thi cung mot phan dau hieu.
3. **Background noise**: nhieu vung anh khong mang thong tin lop, nhung van co the tao ra nhung cap dac trung tuong tu tinh co.

Vi vay, mot prototype toan cuc duy nhat cho moi lop de bi lam loang boi background. Nguoc lai, mot phep ghep cuc bo qua tung token co the bat duoc bang chung tot hon, nhung neu ep moi token phai ghep thi lai qua nhay voi noise.

Do la ly do HROT dung optimal transport khong can bang.

## 2. Nen Tang: Optimal Transport Va Unbalanced Optimal Transport

Optimal transport (OT) co the hieu la bai toan van chuyen khoi luong tu tap token cua query sang tap token cua support sao cho tong chi phi ghep la nho nhat.

Voi query tokens:

```text
X_q = {x_q(1), ..., x_q(T)}
```

va support tokens:

```text
X_s = {x_s(1), ..., x_s(T)}
```

ta lap ma tran chi phi:

```text
C(i,j) = ||x_q(i) - x_s(j)||^2
```

Mot transport plan `P(i,j)` cho biet bao nhieu khoi luong tu query token `i` duoc ghep sang support token `j`.

OT can bang thuong yeu cau tong khoi luong o moi phia phai duoc bao toan. Yeu cau nay khong phu hop voi scalogram noisy, vi no co xu huong ep ca vung background cung phai khop. Unbalanced OT (UOT) noi long rang buoc do. Mo hinh duoc phep van chuyen it khoi luong hon neu chi mot phan anh la bang chung dang tin.

Dang tong quat:

```text
P* = argmin_P <P, C>
     + tau_q KL(P 1 || a)
     + tau_s KL(P^T 1 || b)
     - epsilon H(P)
```

Trong do:

- `C` la chi phi ghep token;
- `a`, `b` la marginal mass cua query va support;
- `tau_q`, `tau_s` dieu khien muc do cho phep lech khoi luong;
- `epsilon` la he so entropy giup giai bang Sinkhorn on dinh;
- `P*` la posterior transport plan sau khi mo hinh da tim cach giai thich query bang support.

Tu goc nhin few-shot, `P*` khong chi la loi giai toi uu. No con la mot ban do bang chung: token nao duoc dung, token nao bi bo qua, va cap ghep nao dang giai thich quyet dinh.

## 3. HROT Ban Dau Giai Quyet Dieu Gi

HROT-FSL co the doc nhu mot dau phan loai few-shot dua tren van tai toi uu. Y tuong cot loi la:

1. Backbone bien moi anh thanh mot luoi token cuc bo.
2. Query duoc so khop voi tung support shot, khong tron het support thanh mot prototype qua som.
3. Chi phi ghep token duoc tinh trong khong gian Euclidean da chuan hoa.
4. Hyperbolic/geodesic statistics duoc dung de du doan transported mass `rho`, tuc la mo hinh nen tin vao bao nhieu bang chung giua query va support shot do.
5. UOT giai bai toan ghep cuc bo voi khoi luong linh hoat.

Trong bien the HROT-H, voi moi query `q`, lop `c`, va support shot `k`, mo hinh tinh:

```text
C_{q,c,k}(i,j) = ||z_q(i) - z_{c,k}(j)||^2
```

Sau do du doan transported mass:

```text
rho_{q,c,k} = EAM(g_{q,c,k})
```

`g_{q,c,k}` gom cac thong ke geodesic:

```text
g = [
  d_H(mu_q, mu_{c,k}),
  d_H(mu_{c,k}, mu_c),
  var_H(q),
  var_H(c,k)
]
```

Y nghia cua bon thong ke nay:

- query co gan support shot do khong;
- support shot do co lech khoi trung tam lop khong;
- query co phan tan bang chung khong;
- support shot co phan tan bang chung khong.

Sau khi co UOT plan `P`, HROT-H tinh:

```text
D = <P, C>
m = sum(P)
```

va shot logit:

```text
logit_{q,c,k} = score_scale * (T * m - D)
```

Trong do `T` la nguong chi phi van chuyen hoc duoc. Cach cham nay co nghia:

```text
Van chuyen nhieu khoi luong chi co loi neu chi phi trung binh moi don vi mass
nho hon nguong T.
```

Day la nen mong quan trong: HROT khong thuong mass mot cach vo dieu kien, ma chi thuong bang chung co chi phi thap.

## 4. Tu HROT-H Den HROT-Q

HROT-H van co mot gia thiet manh:

```text
a_i = rho / T_tokens
b_j = rho / T_tokens
```

Noi cach khac, truoc khi UOT chay, moi token duoc cap khoi luong bang nhau. UOT co the bo bot mass sau do, nhung mo hinh chua noi ro token nao nen duoc xem la foreground evidence va token nao nen di vao background.

HROT-Q duoc thiet ke de sua diem nay. Q them bon co che:

1. **Probe-UOT token reliability**

   Chay mot UOT probe voi marginal gan deu, sau do doc posterior plan de lay cac dac trung nhu row mass, row cost, row entropy, min cost. Token duoc xem la dang tin hon neu no duoc van chuyen nhieu, chi phi thap, va ghep tap trung.

2. **Support cross-shot consensus**

   Trong `K > 1`, support token dang tin hon neu no co token gan tuong ung trong cac support shot cung lop khac. Dieu nay giup giam anh huong cua artifact xuat hien rieng le trong mot shot.

3. **Noise sink**

   Them mot token "sink" de hap thu vung khong khop hoac vung noise. Noise khong bi ep ghep vao token that.

4. **Robust shot pooling**

   Thay vi trung binh cung moi shot, Q hoc cach trong so shot logits de giam anh huong cua support shot xau.

Sau Q, mo hinh da tra loi duoc cau hoi:

```text
Token nao dang tin va shot nao dang tin?
```

Nhung Q van chua tra loi day du cau hoi cau truc:

```text
Cap token duoc ghep co bao toan mau quan he cuc bo cua scalogram khong?
```

Day chinh la khoang trong dan den HROT-R.

## 5. Vi Sao Can HROT-R

Trong scalogram, background noise co the tao ra mot vai cap token co chi phi Euclidean thap. Neu chi nhin tung cap token rieng le, mo hinh co the tin vao nhung cap ghep may rui nay.

Vi du, query co mot vet phong dien that gom nhieu token lien quan voi nhau. Support cung lop cung co vet phong dien voi cau truc tuong tu. Mot cap token dung khong chi nen giong nhau ve dac trung cuc bo, ma con nen co quan he voi cac token xung quanh theo cach tuong thich.

Neu query token `i` ghep voi support token `j`, thi voi cac token query khac `i'` va support token khac `j'` da duoc posterior plan xem la tuong ung, ta mong:

```text
D_q(i, i') gan voi D_s(j, j')
```

Trong do:

- `D_q(i, i')` la khoang cach noi bo giua hai token query;
- `D_s(j, j')` la khoang cach noi bo giua hai token support.

Neu cap `(i,j)` co chi phi Euclidean thap nhung lam vo cau truc noi bo nay, no co the la mot cap ghep tinh co. HROT-R phat hien va phat chi phi cho nhung cap nhu vay.

Ve mat y tuong, day gan voi fused Gromov-Wasserstein: khong chi so sanh dac trung diem-voi-diem, ma con so sanh quan he giua cac diem. Tuy nhien, HROT-R khong giai mot bai toan GW day du vi nhu vay se dat va kho on dinh trong training few-shot. Thay vao do, R dung posterior UOT plan cua Q nhu mot ban do tham chieu roi tinh mot chi phi cau truc nhe hon, on dinh hon.

## 6. Dinh Nghia HROT-R

HROT-R ke thua duong di cua HROT-Q va chi them mot buoc giua adaptive marginal cua Q va final UOT:

```text
HROT-H:
  shot-decomposed UOT
  geodesic EAM rho
  threshold-calibrated score

HROT-Q:
  HROT-H
  + probe-UOT token reliability
  + support consensus
  + noise sink
  + robust shot pooling

HROT-R:
  HROT-Q
  + posterior-derived structural consistency cost
```

Voi moi triple `(q, c, k)`, Q da co:

```text
C        : base Euclidean token cost
rho      : transported mass
a_Q, b_Q : adaptive token marginals cua Q
```

R truoc het chay mot UOT probe khong backprop qua solver:

```text
P_probe = UOT(C, a_Q, b_Q)
```

`P_probe` duoc detach. Dieu nay co chu y: R dung plan nay nhu mot posterior explanation on dinh de do cau truc, khong bat gradient phai chay qua hai tang UOT lien tiep.

Sau do R tinh khoang cach noi bo:

```text
D_q(i,i') = distance giua query token i va query token i'
D_s(j,j') = distance giua support token j va support token j'
```

Cac ma tran nay duoc chuan hoa theo trung binh rieng cua tung anh/support shot de giam phu thuoc vao scale.

Chi phi cau truc cho mot cap ung vien `(i,j)` la ky vong sai khac quan he:

```text
S(i,j)
= E_{(i',j') ~ P_probe} [
    (D_q(i,i') - D_s(j,j'))^2
  ]
```

Dien giai:

- chon cac cap `(i',j')` ma posterior UOT probe cho la dang giai thich nhau;
- kiem tra query token `i` co quan he voi `i'` giong support token `j` co quan he voi `j'` hay khong;
- neu khong giong, cap `(i,j)` bi tang chi phi.

Trong code, cong thuc nay duoc tinh hieu qua hon, khong tao tensor sau chieu:

```text
S = D_q^2 row(P)
  + D_s^2 col(P)
  - 2 D_q P D_s
```

Sau khi chuan hoa `S`, R tao final cost:

```text
C_R = C + lambda_struct * normalize(S)
```

Trong do:

```text
lambda_struct = softplus(raw_structure_cost_weight)
```

`lambda_struct` la tham so hoc duoc, khoi tao boi `--hrot_structure_cost_init`.

Final UOT cua Q sau do duoc giai tren `C_R`, khong phai `C`:

```text
P_R = UOT(C_R with noise sink, a_Q with sink, b_Q with sink)
```

Scoring van theo nguyen tac cua Q/H:

```text
D_R = <P_R_real, C_R>
m_R = sum(P_R_real)
logit_{q,c,k}
  = score_scale * (T_{q,c,k} * m_R - D_R)
```

Sau do robust shot pooling tong hop `K` shot thanh class logit.

## 7. Diem Khac Biet Giua Q Va R

Khac biet ngan gon:

```text
Q hoc token nao dang tin.
R hoc cap ghep token nao vua dang tin vua nhat quan cau truc.
```

Q co the giam mass cua token nhieu, nhung neu mot cap token co chi phi thap va token do co reliability cao, Q van co the dung cap do. R them mot bo loc khac: cap ghep do phai ton trong hinh hoc noi bo duoc posterior plan goi y.

Voi du lieu scalogram, dieu nay co y nghia vi bang chung phong dien khong phai la mot diem rieng le. No thuong la mot mau cuc bo gom nhieu vung lien quan. Background noise co the tao mot diem giong, nhung kho tao ca mot cau truc quan he giong.

## 8. Khac Biet Voi DeepEMD

DeepEMD cung khai thac local matching, nhung cau chuyen thiet ke khac HROT-R.

DeepEMD thuong dua vao:

- cross-reference weighting de tinh trong so token;
- EMD gan can bang hon;
- co che local matching giua query va support representation.

HROT-R dua vao:

- UOT, nen khong ep moi vung phai khop;
- transported mass `rho` thich nghi theo query-class-shot;
- threshold-calibrated score, nen mass chi duoc thuong khi chi phi trung binh du thap;
- noise sink de hap thu bang chung khong khop;
- posterior UOT plan de tao chi phi nhat quan cau truc.

Vi vay, novelty cua R khong phai chi la "them attention token". R dung posterior cua mot bai toan UOT truoc do de tao mot chi phi cau truc cho bai toan UOT cuoi. Day la structural posterior denoising, khong phai cross-reference weighting don thuan.

## 9. Lien He Voi Gromov-Wasserstein Nhung Khong Dong Nhat

Gromov-Wasserstein (GW) so khop hai khong gian diem bang cach bao toan quan he noi bo giua cac diem. Fused GW ket hop chi phi dac trung voi chi phi cau truc.

HROT-R muon giu tinh than nay:

```text
feature match tot + structure match tot
```

Nhung R khong trien khai full GW vi ba ly do:

1. Full GW thuong can toi uu lap phuc tap hon OT thong thuong.
2. Few-shot training can on dinh va chay nhieu episode.
3. HROT da co posterior UOT plan dang tin tu Q, co the dung lam gan dung cau truc.

Do do R la mot thiet ke trung gian:

```text
UOT posterior -> structural mismatch cost -> final UOT
```

No thua huong tinh on dinh cua UOT, dong thoi them mot thanh phan cau truc gan voi GW.

## 10. Thuc Thi Trong Code

Trong `net/hrot_fsl.py`, variant R duoc bat bang:

```python
self.uses_noise_calibrated_transport = variant in {"Q", "R"}
self.uses_structure_consistent_transport = variant == "R"
```

Nghia la R luon di qua branch noise-calibrated transport cua Q, roi them structure cost.

Tham so moi cua R:

```python
self.raw_structure_cost_weight
```

Gia tri duong tuong ung:

```python
self.structure_cost_weight = softplus(raw_structure_cost_weight)
```

Ham trung tam:

```python
_compute_structure_consistency_cost(...)
```

Ham nay nhan:

- `flat_cost`: chi phi Euclidean goc;
- `query_tokens_euc`: token query;
- `flat_support_tokens_euc`: token support theo tung shot;
- `flat_rho`: transported mass;
- `query_mass`, `support_mass`: adaptive token marginals cua Q.

No tra ve:

- `structure_cost`: chi phi cau truc cung shape voi `flat_cost`;
- `structure_probe_mass`: mass cua posterior probe plan.

Trong forward path, R lam:

```text
query/support token marginals cua Q
-> structure_cost
-> final_cost = flat_cost + structure_cost
-> append noise sink
-> final UOT
-> threshold score
-> robust shot pooling
```

Test quan trong:

- `test_hrot_fsl_variant_r_adds_structure_consistent_uot_cost`
- `test_hrot_fsl_variant_r_backpropagates_structure_weight`

Hai test nay kiem tra rang:

- `cost_matrix = base_cost_matrix + structure_cost`;
- `structure_cost` khong am va khac 0;
- `structure_probe_mass` co shape dung;
- `raw_structure_cost_weight` nhan gradient qua loss cuoi.

## 11. Diagnostics Can Xem Khi Danh Gia R

Khi `return_aux=True`, R tra them:

```text
base_cost_matrix
structure_cost
structure_probe_mass
structure_cost_weight
```

Nen xem cung voi cac tensor cua Q:

```text
query_token_mass
support_token_mass
probe_query_reliability
probe_support_reliability
support_consensus
shot_logits
shot_pool_weights
noise_sink_query_mass
noise_sink_support_mass
noise_sink_self_mass
adaptive_transport_cost_threshold
transport_plan
cost_matrix
```

Cach doc diagnostics:

1. Neu `structure_cost` cao tai nhung cap co `base_cost_matrix` thap, R dang phat cac cap ghep re nhung sai cau truc.
2. Neu `structure_cost_weight` sap ve rat nho, structure term co the qua nhieu hoac khong co ich trong training hien tai.
3. Neu `structure_cost_weight` tang va validation tot hon Q, posterior structure dang mang tin hieu co ich.
4. Neu sink mass tang qua manh sau khi them R, chi phi cau truc co the dang qua gat, lam final UOT day qua nhieu bang chung vao sink.

## 12. Thiet Ke Thi Nghiem Va Ablation

Bang ablation nen co:

```text
H
Q
R, structure_cost_init = 0.01
R, structure_cost_init = 0.03
R, structure_cost_init = 0.05
R, structure_cost_init = 0.10
DeepEMD
```

Neu can tach rieng dong gop:

```text
Q full
Q + structure cost, no noise sink
Q + structure cost, no support consensus
R full
```

Cau hoi can tra loi:

1. R co vuot Q khong?
2. Loi ich cua R ro hon o 1-shot hay 5-shot?
3. R co giam cac loi do background noise tao cap token re gia khong?
4. `structure_cost_weight` hoc len hay bi collapse?
5. Cac vung bi phat structure cost co hop ly khi truc quan hoa tren scalogram khong?

Ky vong hop ly:

- R nen tot hon Q khi loi sai cua Q den tu cap token cuc bo re nhung khong nhat quan cau truc.
- R co the khong giup nhieu neu backbone da tao token qua ro, hoac neu noise chu yeu la toan anh thay vi cuc bo.
- R co the can `structure_cost_init` nho neu training dau ky nhay cam.

## 13. Gioi Han Va Rui Ro

HROT-R co mot so gioi han can noi ro trong bai viet:

1. Posterior probe plan duoc detach. Dieu nay tang on dinh, nhung lam structure probe khong duoc hoc end-to-end day du.
2. Structure cost dua tren khoang cach token noi bo trong embedding, khong phai toa do vat ly thoi gian-tan so truc tiep.
3. `lambda_struct` la scalar toan cuc, chua phu thuoc episode hay lop.
4. Neu `P_probe` sai som trong training, chi phi cau truc co the dua tren posterior chua tot.
5. R them chi phi tinh toan so voi Q do can pairwise distance noi bo cho query va support.

Nhung cac rui ro nay cung la ly do thiet ke R duoc giu tuong doi bao thu:

- khong thay backbone;
- khong thay scoring rule;
- khong bo noise sink;
- khong thay UOT solver;
- chi them mot chi phi cau truc co trong so hoc duoc.

## 14. Cach Viet Claim Trong Paper

Claim nen dung:

> HROT-R extends noise-calibrated unbalanced transport with a posterior-derived structural consistency cost. A detached UOT probe provides a soft correspondence distribution, from which the model estimates whether each candidate token match preserves the internal relational geometry of the query and support token sets. The resulting structure cost is added to the Euclidean token cost before the final noise-sink UOT solve.

Khong nen claim qua muc:

- Khong noi R la full Gromov-Wasserstein solver.
- Khong noi R dung hyperbolic distance lam final OT cost.
- Khong noi R thay the DeepEMD bang mot phien ban tuong duong.
- Khong noi structure cost chac chan cai thien moi dataset; can ablation voi Q.

Claim an toan hon:

- R la posterior-UOT structural denoising.
- R them rang buoc cau truc vao matching ma van giu UOT va noise sink.
- R duoc thiet ke cho truong hop bang chung local co cau truc, nhung bi noise tao cap ghep gia re.

## 15. Ket Luan

HROT-R khong phai mot mo hinh moi tach khoi HROT-Q. No la buoc tiep theo tu chuoi lap luan sau:

```text
Few-shot scalogram can local evidence.
Local evidence khong nen bi ep khop toan bo anh.
=> dung UOT.

UOT can biet van chuyen bao nhieu mass.
=> HROT-H dung geodesic EAM va threshold score.

Uniform token mass van qua yeu voi background noise.
=> HROT-Q dung posterior reliability, consensus, sink, shot pooling.

Token reliability van chua dam bao cap ghep bao toan cau truc cuc bo.
=> HROT-R dung posterior UOT plan de tao structure consistency cost.
```

Do do, "R tu dau ra" co the tom gon la:

```text
R sinh ra tu loi con lai cua Q:
Q chon token dang tin, nhung chua phat cap ghep token re ma sai cau truc.
R them chi phi nhat quan cau truc, suy ra tu posterior UOT,
truoc khi giai final noise-sink UOT.
```
