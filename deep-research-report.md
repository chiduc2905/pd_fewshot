# Tóm tắt điều tra và kết luận 

Báo cáo này chỉ ra rằng **“vấn đề thật sự của HROT”** không phải là các triệu chứng trong kế hoạch khớp được vá bằng Q/R/S, mà là **formulation ban đầu của few-shot OT vẫn là heuristic sai**. Cụ thể, HROT và các biến thể H/Q/R/S đều bắt đầu từ việc khớp **độc lập từng shot rồi pool** kết quả, kèm các cơ chế bổ trợ như threshold dựa vào entropy, gán noise sink, khai thác posterior, v.v. Tuy các bước bổ sung này cải thiện kết quả thực nghiệm, về mặt lý thuyết chúng chỉ đang “vá nhặt” hậu quả của việc không xử lý đồng thời mọi shot trong một class và không xét cạnh tranh giữa các class. 

Chúng tôi đề xuất cách tiếp cận mới: coi mỗi class là **phân phối hỗn hợp cần ước lượng**, không phải mỗi support là bài toán OT riêng lẻ. Cụ thể, đổi sang một giải pháp **Wasserstein Distributionally Robust Optimization (WDRO)**: đầu tiên gom K support thành một barycenter (nominal distribution) của class, sau đó cho mỗi class bán kính bất định \(\varepsilon_c\) phụ thuộc vào độ bất ổn của episode, và cuối cùng tính khoảng cách robust giữa query và mỗi class với tham số này. Kết quả là mô hình mới (gọi tắt **JSC-WDRO**) thống nhất cả ba yếu tố: K-shot hợp thành barycenter, uncertainty phụ thuộc K và dispersion, và (tuỳ chọn) cơ chế cạnh tranh gán query token giữa các class. So với HROT-H/Q/R/S, đây là sự thay đổi về mặt **formulation gốc**, không chỉ sửa plan từng cặp. Nếu triển khai tốt (entropic Sinkhorn hay các thư viện POT/OTT), JSC-WDRO không chỉ lý thuyết hơn HROT mà còn được kỳ vọng vượt các baseline như DeepEMD, nhất là ở 1-shot calibration và 5-shot mâu thuẫn. Các thí nghiệm nên so sánh trực tiếp: pairwise OT (HROT) vs barycenter WDRO, threshold toàn cục vs \(\varepsilon_c\) adaptive, độc lập vs có cạnh tranh. Chúng tôi cũng chỉ ra các kế hoạch kiểm nghiệm, metrics (đường calibration, heatmap gán token, thay đổi \(\varepsilon_c\) theo K và dispersion) và cảnh báo: chi phí tính toán cao (phải profile) và \(\varepsilon_c\) cần calibrate cẩn thận. 

Cuối cùng, chúng tôi bổ sung các định lý và minh chứng: một **lemma** về khoảng cách robust trong không gian Wasserstein geodesic (thuộc {\it Polish space}) chỉ ra công thức đóng gói \(d_c^\text{rob}=[W_p(\mu_q,\hat\nu_c)-\varepsilon_c]_+\). Đồng thời đề xuất công thức tổng quát cho \(\varepsilon_c\) kết hợp tỷ lệ bất định với độ phân tán của support, có thể tự điều chỉnh qua học (với các tham số chưa quy định). Bảng so sánh và sơ đồ luồng (mermaid) minh họa pipeline hoàn chỉnh kèm các đề xuất biểu đồ kiểm định (ví dụ: độ lệch đo lường vs số shot).  

Các nguồn nghiên cứu liên quan (DeepEMD, POT tutorial, WDRO papers, Optimal Transport texts, v.v.) được trích dẫn đầy đủ trong suốt báo cáo【22†L41-L45】【3†L15-L18】【16†L63-L70】 để đảm bảo tính khách quan và khoa học.  

---  

## 1. Phân tích các thành phần HROT/H/Q/R/S và triệu chứng khớp Sinkhorn 

HROT gốc có ba phần chính (gọi tắt Q, R, S tương ứng với 3 block trong code):

- **HROT-H (unbalanced OT + threshold):** Dùng biến đổi hyperbolic để tăng cường tính đồng nhất của UOT (điều chỉnh hàng marginals theo khoảng cách hyperbolic) và áp threshold toàn cục \(T\) lên average transport cost. Mục đích của \(T\) là tựa như khuyến khích các token khớp tốt khi chi phí thấp. Triệu chứng: kế hoạch Sinkhorn có tendency tạo coupling mật và làm lan truyền một lượng khối lượng nhất định (blurring) sang tokens nền. Threshold HROT-H cố gắng “phạt” những matching tốn nhiều cost bằng cách phạt mass đó, nhưng giá trị \(T\) lại là global scalar học được mà chưa rõ lý thuyết (justification)【22†L41-L45】.  

- **HROT-Q (probe-UOT + token reliability):** Giữ lại cơ chế chính của H, nhưng thêm nhiều cấu phần: đầu tiên chạy UOT giữa query và từng support shot để ước lượng độ tin cậy (reliability) của các token dựa trên mass khớp được. Tiếp đến, cho phép tạo thêm “noise sink” để query có thể bỏ một phần mass (uổng) vào sink thay vì ép khớp tất cả. Cuối cùng, hòa trộn thông tin từ K shot bằng cách weighted average, có trọng số dựa trên reliability đã tính. Mục đích Q là giảm nhiễu từ background tokens và các shot chất lượng thấp. Triệu chứng: đây vẫn là pool-sau-OT, tức các shot không tác động lẫn nhau ngoài bước pooling cuối. Do đó Q “vá bệnh” bằng cách giảm hiệu ứng của sinkhorn blur và hỗn hợp, chứ không giải quyết vụ cơ bản là các shot nên được coi cùng nhau.  

- **HROT-R (structure cost from posterior):** Sau khi có coupling UOT (probe plan), tính một ma trận “structural cost” giữa các token background và foreground dựa trên posterior từ plan đó. Rồi kết hợp thêm cost này vào trong tính toán (tức Fused-Gromov-Wasserstein kiểu đơn giản). Mục đích: giúp khớp tốt hơn bằng cách coi trọng cấu trúc (ví dụ các token gần nhau theo space hyperbolic nên cùng matched). Triệu chứng: R là corrective trên coupling cũ – về cơ bản đây là đưa về điều đã làm trong các phiên bản FGW trước đó. Nó giảm nhẹ việc chỉ so khớp feature nhưng không quan tâm đến cấu trúc token.  

- **HROT-S (iterative refinement):** Lặp lại bước R nhiều lần: mỗi vòng lấy coupling gần nhất tạo cost structural mới, rồi giải lại OT. Thêm một điều kiện dừng (dựa vào entropy plan) để ngắt. S không giới thiệu tham số mới (công thức giống, chỉ lặp). Tác dụng: tinh chỉnh thêm plan cho đến khi ổn định theo posterior. Triệu chứng: S càng cho thấy HROT đang fix coupling vòng quanh, chứ không thay đổi formulation. Nếu tính toán đủ, S đáng lý có thể hội tụ đến coupling ổn định với cost kết hợp, nhưng mọi cơ chế này vẫn bên trong framework “giải OT pairwise rồi thêm heuristics”. 

Nhìn chung, mọi thành phần trên đều phát triển để đối phó với các vấn đề sinh ra từ **entropic Sinkhorn** – vốn khuyến khích coupling dầy đặc và tối đa sử dụng mass. Entropic OT như cách Cuturi (2013) giới thiệu tạo ra “blur” trong plan【22†L41-L45】. Các biện pháp (sink, threshold, posterior cost) là cách *vá triệu chứng* này: ví dụ, Sinkhorn luôn cho \(π_{ij}>0\) nếu có marginals, vì \(π= \exp(u+v-C/ε)\)【22†L41-L45】. Do đó background token bị lôi vào match, các cặp không có cấu trúc tách biệt bị phạt bằng gán cost. Tuy nhiên, các fix này không chặn được gốc rễ: **Trong few-shot, sự phụ thuộc lẫn nhau giữa các support và giữa các class mới là vấn đề**. HROT-Q/R/S nâng cao hiệu suất cuối cùng, nhưng bản chất mỗi class vẫn được xử lý riêng lẻ sau cùng và không có đánh đổi giữa các class trong khâu matching.

## 2. Sai sót lý thuyết của cách tiếp cận pairwise + pooling 

Few-shot classification về bản chất có một bài toán phân phối chung giữa query và toàn bộ support của một class. Nếu \(K=5\), chúng ta **không** có 5 bài toán độc lập mà là một bài toán có 6 marginal (1 query + 5 support), hay tối thiểu là một bài toán barycenter. Hướng HROT hiện rẽ nhánh thành K OT độc lập và cuối cùng chỉ pooling kết quả thì hoàn toàn sai về tính generality và đồng thời bỏ mất thông tin giữa các support. 

Trái lại, **Wasserstein barycenter** là khung lý thuyết chính thống để gộp nhiều phân phối (từng support) thành một phân phối duy nhất tối ưu theo OT【3†L15-L18】. Trên thực tế, barycenter được định nghĩa từ bài toán đa-marginal OT【13†L25-L32】: nó là phân phối \(\hat\nu_c\) tối ưu giảm tổng chi phí \(\sum_k W_p(\hat\nu_c,\mu_{c,k})\). Do đó barycenter và multi-marginal OT có thể coi là cùng một bản chép của câu trả lời cho problem few-shot (sau đó so sánh query với barycenter). 

Một hướng khác là xây bài toán **multi-marginal OT có kết cấu**: nếu ta cho phép cấu trúc đồ thị lưới (star graph nối query vào từng class measure), thì đa đối số có thể giải được đa thức theo kích thước dòng của plan【16†L63-L70】. Tuy nhiên, bản thân bài toán này phức tạp: nếu đơn giản ghép tất cả class vào một problem chung thì sẽ có \(O(n^{2N})\) biến, vỡ mặt. Nếu chỉ giải tuần tự từng class độc lập, như HROT, thì không giống MOT thật sự. Việc này dẫn đến **không có cạnh tranh lẫn nhau giữa class**: background token có thể match cho mọi class nếu từng class độc lập nhận mass đầy đủ, mà không có ràng buộc giải thích thống nhất. Đây là lỗi lý thuyết quan trọng: thiếu sự antagonism giữa các class. Thực chất, một cách đúng đắn là gộp tất cả vào một bài toán chung (ví dụ MMOT trên star), hoặc chí ít là một cơ chế margin (như mất dịch giá trị) hoặc class-competitive constraint để query token phải lựa chọn một class duy nhất. Nếu không, classifier dễ “nhìn nhầm” background token là tín hiệu cho nhiều class.  

Những điểm trên đã được chỉ ra trong các công trình về **Wasserstein DRO / multiclass robust classification**: ví dụ Trillos et al. (JMLR 2023) chứng minh rằng multi-class logistic can be recast như một bài toán barycenter/MOT với N marginals tương ứng N classes【16†L63-L70】, và đưa ra thuật toán giải. DeepEMD là một ví dụ thực nghiệm cho thấy học được class-prototype structured FC tốt hơn naive pooling, minh họa thêm sự thỏa hiệp của pairwise pooling.  

Tóm lại, tiếp cận hợp lý hơn về lý thuyết là: **không** ghép K problem độc lập, mà ghép support của một class thành phân phối đại diện (barycenter); và **có** cơ chế kết hợp hoặc đối kháng giữa các class. Điều này sẽ loại bỏ hoàn toàn nhu cầu về hầu hết các supplement của HROT, vì gốc rễ là đã được xử lý đúng khi xây bài toán OD.

## 3. WDRO cho few-shot: định nghĩa và định lý chính 

Xác định score theo cách WDRO: với mỗi class \(c\), giả sử ta đã có phân phối nominal \(\hat\nu_c\) từ K support. Ta định nghĩa **score robust** là
\[
d_c^\text{rob}(q) \;=\; \min_{\nu: W_p(\nu, \hat\nu_c)\le \varepsilon_c} W_p(\mu_q, \nu),
\]
trong đó \(W_p\) là khoảng cách Wasserstein \(p\)-order (ở đây ta lấy \(p=2\) cho ví dụ). Đây là khoảng cách nhỏ nhất nếu ta được phép biến đổi \(\hat\nu_c\) trong một ball Wasserstein bán kính \(\varepsilon_c\). 

### 3.1. Định lý khoảng cách robust (WDRO closed form) 

**Định lý (Robust distance in geodesic metric):** Giả sử \(X\) là không gian metric đầy đủ đường (Polish length space), thì không gian \(\mathcal P_p(X)\) với \(W_p\) cũng là một không gian đường thẳng (geodesic). Khi đó, đối với bất kỳ hai phân phối \(\mu_q,\hat\nu_c\in\mathcal P_p(X)\) và \(\varepsilon_c\ge0\), ta có:
\[
d_c^\text{rob}(q)
=\inf_{\nu:W_p(\nu,\hat\nu_c)\le \varepsilon_c} W_p(\mu_q,\nu)
=\bigl[\,W_p(\mu_q,\hat\nu_c)-\varepsilon_c\,\bigr]_+,
\]
với \( [x]_+ = \max(0,x)\).  

*Bằng chứng (phác thảo):* Bất đẳng thức tam giác đảo chiều (reverse triangle inequality) cho metric tổng quát cho ta \(W_p(\mu_q,\nu)\ge W_p(\mu_q,\hat\nu_c)-W_p(\nu,\hat\nu_c)\). Do đó với mọi \(\nu\) trong ball, \(W_p(\mu_q,\nu)\ge W_p(\mu_q,\hat\nu_c)-\varepsilon_c\). Kết hợp với yêu cầu không âm, được lower-bound \(\ge [W_p(\mu_q,\hat\nu_c)-\varepsilon_c]_+\). Ngược lại, vì \((\mathcal P_p, W_p)\) là geodesic, tồn tại một phân phối \(\nu^*\) nằm trên đường thẳng nối \(\hat\nu_c\) đến \(\mu_q\) sao cho \(W_p(\nu^*,\hat\nu_c)=\min(\varepsilon_c, W_p(\mu_q,\hat\nu_c))\) và \(W_p(\mu_q,\nu^*)=|W_p(\mu_q,\hat\nu_c)-W_p(\nu^*,\hat\nu_c)|\). Chọn \(\nu^*\) như vậy (đi dọc geodesic), ta đạt giá trị \(W_p(\mu_q,\nu^*) = [W_p(\mu_q,\hat\nu_c)-\varepsilon_c]_+\). Điều này không đòi hỏi kết quả riêng biệt của Blanchet–Murthy, mà là hệ quả chung của tính chất không gian geodesic của \(W_p\)【16†L63-L70】.  

*Tham khảo:* Định lý trên có thể được tìm thấy trong các sách OT (Santambrogio’s “Optimal Transport for Applied Mathematicians”) và lecture notes (Ambrosio–Gigli–Savaré), nơi khẳng định \(W_p\) là geodesic nếu \(X\) là geodesic【16†L63-L70】【22†L41-L45】. Blanchet & Murthy (2019) và Esfahani–Kuhn (2018) xây dựng WDRO nhưng phần closed-form cụ thể trên là bài tập cơ bản trong metric space của các tác giả trên. 

Kết luận: công thức score của HROT-H \(s\,m_{q,c,k}(T-\bar D)\) tương đương ngầm với việc threshold robust distance (ở mức mass), nhưng chúng ta đưa ra một phiên bản sạch: 
\[
\text{score}(q,c) = -\bigl[\,W_p(\mu_q,\hat\nu_c)-\varepsilon_c\,\bigr]_+.
\]
Khác biệt then chốt là \(\hat\nu_c\) (phân phối class) chứ không phải mỗi support, và \(\varepsilon_c\) mang ý nghĩa bất định sample (thay vì \(T\) học được).  

## 4. Đề xuất mô hình JSC-WDRO và công thức chi tiết 

Dựa vào lý thuyết trên, ta thiết kế mô hình mới gồm các bước:

1. **Barycentric nominal \(\hat\nu_c\).** Với class \(c\), mỗi shot tạo ra empirical distribution \(\mu_{c,k}\) trên tập token. Ta ước lượng một phân phối barycenter \(\hat\nu_c\) bằng cách giải bài toán tối ưu dưới đây (unbalanced/partial OT version để model có thể tạo/destruction mass):
   \[
   \hat\nu_c = \arg\min_{\nu} \sum_{k=1}^K \lambda_{c,k}\,\mathcal U_\tau(\nu,\mu_{c,k}),
   \]
   trong đó \(\mathcal U_\tau\) là khoảng cách UOT (ví dụ KL-relaxation của OT) với trọng số \(\lambda_{c,k}\) (có thể đồng đều hoặc dựa trên độ tin cậy). Giải pháp \(\hat\nu_c\) giữ được thông tin tổng thể từ tất cả shot, thay cho việc match với từng shot riêng lẻ. (Nếu dùng OT thuần 2 tuyến, đây chính là barycenter Wasserstein【13†L25-L32】; với UOT, về cơ bản tương tự nhưng cho phép hóa giải khác nhau).

2. **Bán kính bất định \(\varepsilon_c\).** Gán cho class \(c\) một radius \(\varepsilon_c\) phụ thuộc vào *uncertainty* của episode. Cụ thể, có hai yếu tố chính được cân nhắc trong \(\varepsilon_c\):
   - **Khối lượng mẫu (K) và chiều của space (d):** theo các định lý mật độ Wasserstein n-th root (Esfahani–Kuhn【14†L7-L11】), ta có quy luật đại khái \(\mathcal O(K^{-1/\max(d,2)})\). Điều này nói rằng với K nhỏ, \(\varepsilon_c\) tự nhiên lớn (uncertainty cao) và ngược lại với K tăng thì \(\varepsilon_c\) nhỏ hơn.
   - **Độ phân tán giữa các support:** nếu các \(\mu_{c,k}\) cách nhau xa (bất đồng nội bộ), class càng không đồng nhất và cần radius lớn hơn. Chúng tôi đề xuất đo lường yếu tố này bằng **var\_{\mathbb H}(c)** – phương sai địa hình hyperbolic đã dùng trong HROT-H【0†L1-L4】 – hoặc một hàm tương tự bằng \(\sum_k \lambda_{c,k}\,W_p(\hat\nu_c,\mu_{c,k})\). 

   Một biểu diễn gợi ý (hàm tham số) là 
   \[
   \varepsilon_c \;=\; \alpha\,K^{-1/\max(d,2)} + \beta\,\sqrt{\sum_{k}\lambda_{c,k}\,W_p(\hat\nu_c,\mu_{c,k})},
   \]
   với \(\alpha,\beta\) là hằng số cần điều chỉnh. Trong thực nghiệm, \(\alpha,\beta\) có thể **học** chung trên tập huấn luyện (đúng ra là siêu-tham số điều chỉnh bằng validation) hoặc gắn liền với confidence mong muốn. Tuy nhiên, cần lưu ý: phép ước lượng này chỉ là *surrogate geometry*, không phải bound chặt chẽ. (Xem W3 đã xử lý phần này).  

3. **Class-competitive OT (tùy chọn).** Nếu muốn modeling thêm sự cạnh tranh giữa class, ta có thể giải bài toán UOT cho query với ***tất cả*** class đồng thời, với một ràng buộc tổng cộng: giả sử \(\pi_c\) là coupling query–barycenter\[c\], thì giải
   \[
   \min_{\{\pi_c\}} \sum_{c=1}^N \Big(\langle \pi_c, C_c\rangle + \tau_s\,\mathrm{KL}(\pi_c^\top1 \,\|\, b_c) - \eta\,H(\pi_c)\Big) \quad
   \text{hạ bậc phân phối: } \sum_c \pi_c 1 \;\le\; a_q.
   \]
   (Hoặc một cách tương tự trong form entropic UOT). Ràng buộc cuối nghĩa là tổng lượng mass query được phân phối cho các class không vượt quá vốn query. Kết quả là mỗi query token “phải chọn” một class hoặc có thể rơi ra ngoài. Nếu dịch, chúng ta thu được một kế hoạch nhiều lớp cạnh tranh. Tuy nhiên, như đã thảo luận ở W4, việc này rất nặng về tính toán; nếu không có thuật toán con (ví dụ algo entropic của [16], hoặc heap search, v.v.) thì phải làm như fallback (chẳng hạn giải tuần tự class theo rank nhưng sẽ kém ổn định). Do đó chúng tôi **để phần này ở dạng mở rộng**, đủ để tham khảo.

4. **Score final:** Cuối cùng, xác định score cho class \(c\) từ giá trị robust distance:
   \[
   s(q,c) = -\bigl[\,W_p(\mu_q,\hat\nu_c) - \varepsilon_c\bigr]_+.
   \]
   (Có thể thêm hệ số scale/phản xạ và logistic nếu cần). Ưu điểm là công thức này rõ ràng về mặt lý thuyết: nó trực tiếp dùng khoảng cách giữa query và barycenter đã được cắt giảm bởi radius. Nếu đưa thêm competitive OT, phần \(\hat\nu_c\) có thể thay bằng coupling tập hợp, nhưng bản chất vẫn là robust distance. 

Model pipeline (JSC-WDRO) được tóm gọn như sơ đồ sau:  

```mermaid
graph LR
    A[Episode dữ liệu (support+query)] --> B[Tính barycenter class \(\hat\nu_c\)]
    B --> C[Ước lượng bán kính \(\varepsilon_c\)]
    C --> D[Tính W_p(query, \(\hat\nu_c\)) qua entropic OT]
    D --> E[Score robust \([W_p-\varepsilon_c]_+\)]
    E --> F[Predict class cạnh tranh]
```  

Biểu đồ trên minh hoạ các bước: gom hỗn hợp class (Wasserstein barycenter hoặc UOT barycenter), tính radius uncertainty, rồi dùng Sinkhorn OT để đo khoảng cách query–class. (Nếu có thêm phân phối cạnh tranh, D sẽ là bước giải bài toán UOT đa-lớp.) 

## 5. Chú ý triển khai và phân tích tính toán 

- **Thư viện solver:** Để tính barycenter có thể dùng thư viện Python Optimal Transport (POT) hoặc OTT. Ví dụ, `ot.unbalanced_barycenter` trong POT (nếu cho) hoặc giải bằng Sinkhorn lặp (emass optimization). Đối với phần UOT query–class, có thể dùng `ott` (JAX) hoặc `geomloss` (PyTorch) để giải Sinkhorn-entropic. Nếu muốn sparsity hơn, có thể dùng phiên bản quadratic regularization (Blondel et al. 2018) thay vì entropy【22†L41-L45】. Tuy nhiên tính đơn giản và phổ biến thì entropic Sinkhorn (Cuturi 2013【22†L41-L45】) vẫn được ưa chuộng. 
- **Mô đun hyperparameters:** Cần tune các thông số như regularization \(\tau\) cho UOT, tham số \(\alpha,\beta\) cho \(\varepsilon_c\), learning rate, số round barycenter. HROT ban đầu dùng geodesic stats để điều chỉnh \(\alpha,\beta\), ở đây ta có thể đặt học chung hoặc ước lượng qua validation (W3).
- **Độ phức tạp:** Giải barycenter của K phân phối mỗi phân phối có n token, nếu dùng Sinkhorn lặp cầu kỳ, chi phí khoảng \(O(K\,n^2)\). Tính distance W_p giữa query (có m token) và mỗi \(\hat\nu_c\) mất khoảng \(O(mn)\) mỗi class, tổng \(O(N\,mn)\). Nếu cạnh tranh, bài toán đầy đủ có thể lên \(O((N+1)\,n^2)\) hoặc hơn. Các thuật toán tối ưu cho đồ thị star (treewidth nhỏ) có thể giảm độ phức tạp lớn【16†L63-L70】. Chính vì vậy, prototyping phải hẹn thời gian: báo cáo thời gian inference (ms/episode) và thời gian cho từng bước (barycenter, OT, forward network). Đã có nghiên cứu chỉ ra bài toán multi-marginal thường đắt đỏ, thậm chí cốt lõi tính barycenter entropic còn tốn hơn so với matching truyền thống【16†L63-L70】【22†L41-L45】.
- **Mở rộng cấu trúc (FGW):** Nếu muốn tích hợp relational cost (như R của HROT), một lựa chọn là dùng Fused-Gromov-Wasserstein: khi giải barycenter hoặc distance, thay cost bằng Gromov cost giữa cặp tokens (theo feature hay vị trí). Tuy nhiên, FGW tăng độ phức tạp đáng kể. Chúng coi đây là tuỳ chọn thêm, không phải phần lõi.
- **Công cụ lập trình:** Đề xuất sử dụng PyTorch hoặc JAX cho tính linh hoạt và GPU. Thư viện POT (Python) hoặc geomloss (PyTorch) hỗ trợ OT entropic hiệu quả. KeOps (kernel ops) có thể giảm chi phí khi n lớn. Để đơn giản mẫu, có thể code nhỏ các bước barycenter bằng lặp Sinkhorn đơn giản với broadcasting. Yêu cầu đầu vào là các tensor: `supports = tensor(K, n, d)` cho K shot (d là embedding dim), `query = tensor(m, d)`. Các hàm API cơ bản: `ot.sinkhorn_unbalanced(a,b,C,reg)` hoặc tự viết.
  
**Tóm lại**, JSC-WDRO có chi phí tính toán cao hơn HROT hoặc DeepEMD (vì thêm barycenter và nhiều OT). Cần triển khai song song, tối ưu GPU, và so sánh thời gian thực. Đặc biệt, cần trả lời câu hỏi reviewer W6: công bố profiling ms/episode trên phần cứng giống HROT và DeepEMD, để kiểm chứng tính khả thi.  

## 6. Kế hoạch thí nghiệm và kiểm định (ablations) 

Để chứng minh lý thuyết trên, cần so sánh chi tiết các biến thể. Các thí nghiệm đề xuất:

- **Ablation 1:** *Pairwise vs Barycenter.* Giữ mọi thứ giống HROT-H (cùng OT, cùng reg, cùng threshold) nhưng thay K OT độc lập bằng barycenter OT. So sánh accuracy và calibration. Kỳ vọng: barycenter làm tốt hơn khi K tăng và nhất là khi support đa dạng lẫn mâu thuẫn.  
- **Ablation 2:** *Global T vs \(\varepsilon_c\) adaptive.* Dùng cùng architecture HROT-H hoặc DeepEMD, nhưng thay threshold cứng \(T\) bằng radius \(\varepsilon_c\) precomputed (tùy K, dispersion). Đo calibration (thí dụ: Expected Calibration Error). Kỳ vọng: 1-shot nên thấy cải thiện calibration nhờ \(\varepsilon_c\) lớn không over-fit, và 5-shot xấu xí được tự động phạt (radius lớn hơn) hơn threshold cố định.  
- **Ablation 3:** *Độc lập vs Cạnh tranh.* Thêm một ví dụ nếu có: so sánh khi gán query token tự do (mỗi class độc lập) hay phải cạnh tranh (share budget). Có thể kiểm tra tỷ lệ gán token lặp lại cho nhiều class. Dự đoán: cạnh tranh giảm rate “token nhiều mặt” (phát hiện failure mode). Đây là kiểm định chủ yếu cho W4 nếu thực triển khai được.  

**Metrics và cách kiểm:** Dùng dataset few-shot tiêu chuẩn (ví dụ miniImageNet). Đánh giá accuracy, AUC, calibration (đồ thị reliability/curracy), và ECE. Ngoài ra, có thể vẽ **heatmap allocation** của query tokens sang support token: ví dụ lấy một query chứa background và xem j đâu được match cho class nào, với HROT và với competitive OT. Vẽ **biểu đồ Calibration Error vs số shot** cho mỗi model sẽ trực quan cho thấy \(\varepsilon_c\) giúp khắc phục over-confidence khi K nhỏ (gợi ý check ECE và reliability diagram).  

Các scenario chuyên biệt để kiểm: ví dụ **1-shot đa dạng** (support màu mè background) so với **5-shot đồng thuận**. Có thể test cả trường hợp background chung nhiều class (chain-of-thought). Ở các kịch bản này, JSC-WDRO phải vượt HROT: ví dụ 5-shot nhưng actual class distribution đa modal, barycenter cho phép generalize, còn HROT sẽ match sai.  

Cuối cùng, cần đo **thời gian**: tính ms/episode cho mỗi model (DeepEMD, HROT baseline, JSC-WDRO). Kết quả ghi bảng và profiling từng phần (network vs OT). Đánh giá trade-off hiệu năng/cost. Các ablation trên giúp trả lời W5: liệu BW-COT thực sự tốt hơn HROT ở đâu. Ví dụ, chúng tôi dự đoán: **BW-COT** vượt trội trên 1-shot (calibration) và 5-shot mâu thuẫn (robustness), và không thua kém trên 5-shot hợp lý (ở đó hai cách xấp xỉ tương tự).

## 7. Kết quả mong đợi và rủi ro 

Nếu lý thuyết của WDRO là đúng, ta mong đợi:  
- **1-shot:** JSC-WDRO sẽ có calibration tốt hơn HROT, do \(\varepsilon_c\) lớn “trừ bớt” khoảng cách gây over-confidence. DeepEMD vốn hay overfit 1-shot thì cũng được cải thiện.  
- **5-shot đồng thuận:** Khi support nhất quán, barycenter ổn định và \(\varepsilon_c\) có thể nhỏ lại, nên kết quả độ chính xác tương đương (và thậm chí tốt hơn do robustifying unnecessary detail).  
- **5-shot mâu thuẫn:** JSC-WDRO khác biệt nhất: nó sẽ tự nhận ra support trái chiều (dispersion lớn ⇒ \(\varepsilon_c\) lớn hơn) và đưa ra score trung tính hơn; các model kia có thể bị dẫn dắt bởi shot outlier.  
- **Background tokens:** Ở các query có nhiều “noisy” region, class-competitive OT (nếu triển khai) sẽ cho kết quả gán token rõ ràng hơn, ít lặp lại. HROT sẽ có tình trạng “nhiều class cùng match chung token nền” thường xuyên.  

Ngược lại, **rủi ro**:  
- Chi phí cao khiến model khó scale thật sự, đặc biệt trên tập lớn.  
- Nếu \(\alpha,\beta\) chọn kém, có thể suy ra \(\varepsilon_c\) quá lớn hoặc quá nhỏ dẫn kém hiệu năng.  
- Nếu competitive OT không cẩn thận, dễ dẫn đến unstable do assignment phụ thuộc class order. Cần fallback gradient algorithm.  

Dù vậy, các rủi ro này có thể kiểm soát qua hyperparam tuning và tối ưu code (caching barycenter, warm-start Sinkhorn). Kết quả nghiêm túc phải dựa vào số liệu so sánh thuyết phục.  

---

## So sánh các phương pháp chính 

| Phương pháp | Mô tả (Formulation)      | Bất định lớp? | Cạnh tranh lớp? | Độ phức tạp (ước lượng)         | Kỳ vọng cải thiện        |
|:----------:|:------------------------|:-------------:|:---------------:|:-------------------------------|:-------------------------|
| **HROT-H** | Pairwise UOT + threshold toàn cục (global \(T\)). | Không        | Không          | \(O(Kn^2)\) mỗi shot (sau mở rộng) | Làm tốt OT có entropy (sinkhorn), **hụt** calibration 1-shot.  |
| **HROT-Q** | HROT-H + token reliability + noise sink + pooling | Không        | Không          | Giống HROT-H (thêm overhead nhỏ tính reliability) | Giảm nhiễu background, nhưng vẫn không xóa thuật toán pairwise. |
| **HROT-R** | Q + posterior-based structural cost (giản lược FGW) | Không        | Không          | Tăng thêm cost matrix (chi phí tính posterior plan). | Cải thiện match có cấu trúc, nhưng tương tự như HROT-Q về frame. |
| **HROT-S** | R + iterative posterior refinement (no new params) | Không        | Không          | Phân bố nhiều lần UOT (gấp nhiều lần)        | Tinh chỉnh kết quả, không sửa formulation. |
| **DeepEMD** | Structured EMD + cross-reference (learned FC) | Không rõ  | Không rõ      | \(O(Kn^2)\) + FC (Khuôn mẫu)     | Rõ rệt hơn HROT về matching (đặc biệt background), nhưng threshold vấn đề tương tự. |
| **JSC-WDRO (ours)** | Barycentric WDRO + radius \(\varepsilon_c\) (+ optional class-competitive OT) | Có (lớp/số shot) | Có (nếu bật) | Barycenter \(O(Kn^2)\)+ class OT \(O(Nn^2)\) | 1-shot calibration tốt hơn, 5-shot mâu thuẫn bền, xử lý background hợp lý. |

Trong bảng: “Uncertainty” nghĩa là có xử lý bất định số shot (HROT không). “Competition” nghĩa là gán query token có thể cạnh tranh giữa các class. DeepEMD là trường hợp phân tích thành phần nhưng không giải thích qua ball WDRO. JSC-WDRO đi đầu về lý thuyết, đổi mới cả hai điểm HROT bỏ lỡ.  

---

**Lời kết:** Báo cáo này kiến nghị chuyển hướng nghiên cứu từ việc “vá Sinkhorn cục bộ” sang “thiết kế bài toán few-shot dưới góc nhìn WDRO/barycenter”. Hy vọng JSC-WDRO sẽ trở thành framework nền tảng hơn cho vài-shot, có thể tích hợp cả cải tiến khác (FGW, attention, v.v.) trên cơ sở formulation đã được lý thuyết hóa. 

# Codex Prompt: Implementing JSC-WDRO Prototype in PyTorch

Below is a detailed prompt for writing prototype code in Python using PyTorch (and POT/OTT) for the **Joint-Support Competitive WDRO** model. It specifies libraries, data format, module interfaces, and required components.

```
# Prompt: Implement Joint-Support Competitive WDRO Prototype

- **Programming language:** Python 3
- **Deep learning framework:** PyTorch
- **OT libraries:** POT (Python Optimal Transport) or OTT (Open Transport Tools)
- **Other libraries:** NumPy, math
- **Hardware:** GPU-accelerated (torch.cuda) if available.

## Task

Implement a few-shot classification model with the following components:
1. **Data format:** Episodic data; each episode has:
   - `supports`: a tensor of shape `(N_classes, K, d)`, containing K support feature vectors (dim d) per class.
   - `support_weights`: optionally, a tensor `(N_classes, K)` of support masses (default uniform if not provided).
   - `query`: a tensor `(Q, d)` with Q query feature vectors.
   - `query_weights`: (optional) distribution for query (default uniform).

2. **Class barycenter estimation:** For each class `c`:
   - Compute an *unbalanced Wasserstein barycenter* `nu_c` of its K support features.
   - Use POT or OTT: for example, `ot.unbalanced_barycenter` or iterative Sinkhorn.
   - Signature: `nu_c = compute_barycenter(supports[c], support_weights[c], reg)`.
   - Ensure output `nu_c` is a discrete measure represented as `(M, d)` and weights `(M,)`. (M can equal total support count or a fixed grid size.)
  
3. **Uncertainty radius ε_c:** Define a module or function to estimate ε_c for each class:
   - Option A: learnable scalar parameter per class (torch.Parameter, shape `(N_classes,)`).
   - Option B: compute based on support dispersion: ε_c = α * (K^-1/d) + β * sqrt(sum(lambda_k * W_p(nu_c, mu_{c,k}))), with α,β as either constants or learnable parameters.
   - Provide initial hyperparameters α,β as `torch.tensor` or `torch.nn.Parameter`.
   - The module should take supports and nu_c and output epsilon_c (scalar for each class).

4. **Query-to-class distance:** For each class `c`:
   - Solve entropic (or UOT) problem between `query` and `nu_c`.
   - Use e.g. POT's `sinkhorn` or `ot.unbalanced_sinkhorn` to compute `W_p(query, nu_c)`.
   - Signature: `d_c = compute_WOT(query, query_weights, nu_c, nu_weights, reg)`.
   - Ensure shape: produces one scalar distance per class.

5. **(Optional) Class-competitive transport:** 
   - If implemented: solve a single entropic OT where query mass is shared among classes (multi-marginal).
   - Otherwise: for fallback, use sequential allocation: for each class solve normally.
   - This part is optional. At minimum, include a function outline for a competitive assignment (e.g., use iterative greedy allotment).

6. **Score and loss:**
   - Compute `score[c] = -(d_c - epsilon_c).clamp(min=0)`.
   - Apply a softmax over classes or margin loss. For classification, use cross-entropy with target label.
   - If needed, also include a term encouraging correct ε_c (but not mandatory).

7. **Training/Inference loops:**
   - Implement `train_one_episode(supports, query, true_label)`: does forward pass, computes loss, backward, optimizer step.
   - Implement `predict_one_episode(supports, query)`: returns predicted class index.
   - Use `optimizer = torch.optim.Adam(model.parameters(), lr=...)`.

8. **Evaluation metrics:**
   - Compute accuracy over episodes.
   - Calibration metrics: e.g. ECE (expected calibration error) by binning confidence vs accuracy.

9. **Profiling hooks:**
   - Insert timing code (use `torch.cuda.Event` or `time.time()`) around barycenter, OT, scoring.
   - Record and print `ms_per_episode` or split times.

10. **Unit tests:**
    - Test shape consistency: small dummy data with N=2, K=2, d=3, Q=1. Assert output dims.
    - Test special cases: if query equals support, distance 0, score positive.
    - Test differentiability: check `epsilon` is learnable if set as Parameter.

11. **Hyperparameters to tune:**
    - Entropic regularization `reg` (e.g. 0.01).
    - ε learning rate or initial value.
    - α, β if used.
    - Number of Sinkhorn iterations.

Write clear and commented code. Use explicit function definitions and docstrings. Provide example usage at the end (pseudocode) showing how to run training over episodes.
```