# Phân tích kết quả benchmark

## Kết quả tổng quan

Benchmark dùng cùng một bộ dữ liệu cho Baseline Agent và Advanced Agent. Kết quả
gần nhất được lưu tại `state/benchmark_results.md` và
`state/benchmark_results.json`.

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1809 | 16132 | 0.000 | 0.000 | 0 | 0 |
| Advanced | 2602 | 31718 | 0.929 | 0.929 | 3088 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 493 | 25269 | 0.000 | 0.000 | 0 | 0 |
| Advanced | 677 | 20092 | 1.000 | 1.000 | 4253 | 17 |

## Vì sao Advanced recall tốt hơn Baseline?

Baseline chỉ giữ lịch sử trong cùng một `thread_id`. Các câu recall được đặt ở
thread mới, vì vậy Baseline không còn dữ liệu của những thread huấn luyện trước
đó. Kết quả recall của Baseline gần bằng 0 là đúng với thiết kế short-term-only.

Advanced trích xuất các fact bền vững như tên, nghề nghiệp, nơi ở, sở thích và
phong cách trả lời vào `User.md`. File này được đọc lại theo `user_id`, không phụ
thuộc vào `thread_id`, nên agent vẫn recall được thông tin khi bắt đầu session
mới. Cơ chế upsert cũng cho phép thông tin mới, ví dụ nghề MLOps engineer hoặc
nơi ở Huế, thay thế fact cũ.

## Vì sao Advanced tốn hơn ở hội thoại ngắn?

Trong Standard Benchmark, Advanced xử lý 31718 prompt tokens, cao hơn 16132 của
Baseline. Mỗi lượt của Advanced phải mang thêm nội dung `User.md` và metadata
memory. Các thread ngắn chưa đủ dài để kích hoạt compact, nên chi phí bổ sung
này chưa được bù lại bằng việc nén lịch sử.

Đây là trade-off trực tiếp: Advanced trả thêm chi phí prompt và agent tokens để
đổi lấy cross-session recall 0.929, trong khi Baseline rẻ hơn nhưng recall chỉ
đạt 0.000.

## Compact tối ưu prompt tokens như thế nào?

Trong stress benchmark, Baseline liên tục đưa toàn bộ lịch sử dài vào prompt,
làm tổng prompt tokens tăng lên 25269. Advanced compact 17 lần: các message cũ
được tóm tắt, chỉ summary và một số message gần nhất được giữ trong context.
Nhờ vậy Advanced chỉ xử lý 20092 prompt tokens, giảm 5177 tokens, tương đương
khoảng 20.5% so với Baseline trong phép đo này.

Compact chủ yếu tối ưu `Prompt tokens processed`, không đảm bảo `Agent tokens
only` luôn thấp hơn. Advanced vẫn sinh nhiều token hơn vì phải xác nhận memory và
trả lời đầy đủ các fact đã recall. Summary cũng có nguy cơ làm mất chi tiết, nên
các fact bền vững quan trọng được tách riêng vào `User.md` thay vì chỉ đưa vào
compact summary.

## Memory growth và rủi ro

Baseline không có persistent memory nên memory growth bằng 0. Advanced tăng 3088
bytes ở benchmark chuẩn và 4253 bytes ở stress benchmark. Mức tăng này nhỏ trong
bộ dữ liệu lab, nhưng có thể tích lũy khi số user và số fact tăng lên.

Các rủi ro chính gồm:

- lưu nhầm câu hỏi, câu đùa hoặc thông tin tạm thời thành fact bền vững;
- giữ đồng thời fact cũ và fact mới khi người dùng đính chính;
- `User.md` phình to, làm tăng prompt tokens ở mỗi lượt;
- summary compact có thể làm mất chi tiết quan trọng của hội thoại cũ.

Hệ thống giảm các rủi ro này bằng extractor bảo thủ, canonical key để cập nhật
fact, loại trùng lặp và tách persistent memory khỏi compact memory. Trong hệ
thống production, nên bổ sung confidence threshold, memory decay, giới hạn kích
thước file và test conflict handling rộng hơn.

## Kết luận

Kết quả thể hiện đúng câu chuyện của bài lab: Baseline đơn giản và rẻ trong ngữ
cảnh ngắn nhưng không recall qua session; Advanced tốn thêm chi phí memory nhưng
nhớ fact bền vững tốt hơn; khi hội thoại đủ dài, compact memory giúp giảm prompt
load mà vẫn giữ các fact quan trọng trong `User.md`.

## Bonus memory guardrails

Bản nâng cao bổ sung bốn guardrail:

- mỗi entity được biểu diễn bằng key, value, category, confidence và source;
- chỉ fact đạt `MEMORY_CONFIDENCE_THRESHOLD` mới được ghi vào `User.md`;
- câu hỏi tìm kiếm thông tin không bị xem nhầm là fact của người dùng;
- correction overwrite canonical fact cũ, còn fact quá cũ giảm confidence theo
  `MEMORY_DECAY_DAYS`.

Metadata phục vụ confidence và decay nằm trong `User.meta.json` cạnh
`User.md`. Cách tách này giữ cho hồ sơ Markdown dễ đọc, đồng thời cho phép
runtime lọc fact cũ trước khi đưa vào prompt.
