"""
Bộ kiểm thử tự động (Test Suite) đánh giá tính đúng đắn của Model Machine Learning
Các nhóm bài test:
1. Test Cấu trúc & Tính toàn vẹn dữ liệu (Data Integrity & Output Shape)
2. Test Tính chất Phân bổ Xác suất (Probability Calibration)
3. Test Ngữ cảnh & Nghiệp vụ Thực tế (Domain / Semantic Correctness)
4. Test Ngưỡng Hiệu năng Tối thiểu (Performance Benchmark Thresholds)
5. Test Độ bền & Trường hợp Biên (Robustness & Edge Cases)
6. Test Tính Tái lặp & Bất biến (Determinism)
"""

import sys
for stream in [sys.stdout, sys.stderr]:
    if stream and stream.encoding != 'utf-8':
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from netflix_ml_model import (
    load_data,
    split_data,
    train_model,
    VALID_CLASSES
)

class TestNetflixModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Khởi tạo và huấn luyện mô hình 1 lần dùng chung cho toàn bộ test suite."""
        print("\n" + "=" * 70)
        print("  BẮT ĐẦU THIẾT LẬP TEST SUITE: LOAD DỮ LIỆU & HUẤN LUYỆN MODEL")
        print("=" * 70)
        cls.X, cls.y, _ = load_data("netflix_titles_cleaned.csv")
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = split_data(
            cls.X, cls.y, test_size=0.20, random_state=42
        )
        cls.model = train_model(cls.X_train, cls.y_train)
        cls.classes = list(cls.model.named_steps['classifier'].classes_)
        print(f"[+] Model đã sẵn sàng với {len(cls.classes)} nhóm nhãn: {cls.classes}")
        print("=" * 70 + "\n")

    # =========================================================================
    # NHÓM 1: KIỂM TRA TÍNH TOÀN VẸN ĐẦU RA (OUTPUT INTEGRITY)
    # =========================================================================
    def test_01_output_shape_matches_input(self):
        """Kiểm tra số lượng dự đoán phải khớp chính xác số lượng mẫu đưa vào."""
        sample_inputs = self.X_test.iloc[:50]
        preds = self.model.predict(sample_inputs)
        self.assertEqual(
            len(preds), len(sample_inputs),
            f"Số lượng dự đoán ({len(preds)}) không khớp số lượng mẫu đầu vào ({len(sample_inputs)})"
        )

    def test_02_no_null_or_nan_predictions(self):
        """Kiểm tra kết quả dự đoán không chứa giá trị null, NaN hoặc rỗng."""
        preds = self.model.predict(self.X_test)
        self.assertFalse(
            pd.isna(preds).any(),
            "Kết quả dự đoán có chứa phần tử NaN hoặc None!"
        )
        self.assertTrue(
            all(bool(str(p).strip()) for p in preds),
            "Kết quả dự đoán có chứa chuỗi rỗng!"
        )

    def test_03_predicted_labels_are_valid(self):
        """Kiểm tra toàn bộ nhãn dự đoán phải nằm trong tập nhãn hợp lệ."""
        preds = self.model.predict(self.X_test)
        unique_preds = set(preds)
        invalid_labels = unique_preds - set(VALID_CLASSES)
        self.assertEqual(
            len(invalid_labels), 0,
            f"Phát hiện nhãn dự đoán không hợp lệ: {invalid_labels}"
        )

    def test_04_single_string_prediction_support(self):
        """Kiểm tra mô hình có thể dự đoán linh hoạt trên 1 mẫu đơn lẻ dạng list hoặc Series."""
        single_input = ["Movie | Children & Family Movies | Cute animals play in the forest."]
        pred = self.model.predict(single_input)
        self.assertEqual(len(pred), 1)
        self.assertIn(pred[0], VALID_CLASSES)

    # =========================================================================
    # NHÓM 2: KIỂM TRA TÍNH CHẤT PHÂN BỔ XÁC SUẤT (PROBABILITY CALIBRATION)
    # =========================================================================
    def test_05_probabilities_range(self):
        """Kiểm tra xác suất trả về từ predict_proba phải luôn nằm trong khoảng [0, 1]."""
        sample_inputs = self.X_test.iloc[:100]
        probs = self.model.predict_proba(sample_inputs)
        self.assertTrue(
            np.all(probs >= 0.0) and np.all(probs <= 1.0),
            "Có giá trị xác suất nằm ngoài phạm vi [0.0, 1.0]!"
        )

    def test_06_probabilities_sum_to_one(self):
        """Kiểm tra tổng xác suất các lớp trên mỗi dòng phải xấp xỉ 1.0 (dung sai 1e-5)."""
        sample_inputs = self.X_test.iloc[:100]
        probs = self.model.predict_proba(sample_inputs)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, rtol=1e-5, atol=1e-5,
            err_msg="Tổng xác suất các lớp trên một dòng không bằng 1.0!"
        )

    def test_07_prediction_matches_argmax_probability(self):
        """Kiểm tra nhãn trả về từ predict() phải trùng khớp với lớp có xác suất cao nhất."""
        sample_inputs = self.X_test.iloc[:100]
        preds = self.model.predict(sample_inputs)
        probs = self.model.predict_proba(sample_inputs)
        
        expected_indices = np.argmax(probs, axis=1)
        expected_labels = [self.classes[i] for i in expected_indices]
        
        self.assertListEqual(
            list(preds), expected_labels,
            "Nhãn predict() không đồng nhất với argmax của predict_proba()!"
        )

    # =========================================================================
    # NHÓM 3: KIỂM TRA NGỮ CẢNH & NGHIỆP VỤ THỰC TẾ (SEMANTIC ACCURACY)
    # =========================================================================
    def test_08_semantic_kids_family_content(self):
        """Kiểm tra nội dung hoạt hình thiếu nhi phép thuật phải phân loại là 'Kids & Family'."""
        kids_film = [
            "Movie | Children & Family Movies | A magical fairy and her cute animal friends sing songs and go on a fun adventure to help a princess."
        ]
        pred = self.model.predict(kids_film)[0]
        probs = self.model.predict_proba(kids_film)[0]
        confidence = np.max(probs) * 100
        
        self.assertEqual(
            pred, "Kids & Family",
            f"Kỳ vọng 'Kids & Family' nhưng mô hình dự đoán '{pred}' (Độ tin cậy: {confidence:.2f}%)"
        )
        self.assertGreater(confidence, 80.0, "Độ tin cậy cho phim trẻ em phải > 80%")

    def test_09_semantic_adults_content(self):
        """Kiểm tra nội dung bạo lực, ma túy, giết người hàng loạt phải phân loại là 'Adults (18+)'."""
        adult_film = [
            "TV Show | Crime TV Shows, International TV Shows | A brutal drug cartel assassin carries out violent murders and bloody revenge against rival mafias."
        ]
        pred = self.model.predict(adult_film)[0]
        probs = self.model.predict_proba(adult_film)[0]
        confidence = np.max(probs) * 100

        self.assertEqual(
            pred, "Adults (18+)",
            f"Kỳ vọng 'Adults (18+)' nhưng mô hình dự đoán '{pred}' (Độ tin cậy: {confidence:.2f}%)"
        )
        self.assertGreater(confidence, 80.0, "Độ tin cậy cho phim người lớn 18+ phải > 80%")

    def test_10_semantic_teens_content(self):
        """Kiểm tra nội dung học đường tuổi teen, dạ hội prom phải phân loại là 'Teens (13-14+)'."""
        teen_film = [
            "Movie | Comedies, Romantic Movies | High school teenagers deal with school gossip, dating crushes, and preparing for the senior prom night."
        ]
        pred = self.model.predict(teen_film)[0]
        self.assertEqual(
            pred, "Teens (13-14+)",
            f"Kỳ vọng 'Teens (13-14+)' nhưng mô hình dự đoán '{pred}'"
        )

    # =========================================================================
    # NHÓM 4: KIỂM TRA NGƯỠNG HIỆU NĂNG TỐI THIỂU (PERFORMANCE THRESHOLDS)
    # =========================================================================
    def test_11_overall_accuracy_benchmark(self):
        """Kiểm tra Mean Accuracy trên tập kiểm thử phải đạt ngưỡng tối thiểu >= 60%."""
        accuracy = self.model.score(self.X_test, self.y_test)
        # Đoán ngẫu nhiên 3 nhóm chỉ đạt ~33%, baseline mô hình cần đạt >= 60%
        MIN_ACCURACY = 0.60
        self.assertGreaterEqual(
            accuracy, MIN_ACCURACY,
            f"Accuracy của mô hình ({accuracy:.4f}) không đạt ngưỡng tối thiểu ({MIN_ACCURACY})!"
        )

    def test_12_per_class_metrics_benchmark(self):
        """Kiểm tra Recall của nhóm Adults (18+) >= 0.70 và Precision của Kids >= 0.75."""
        y_pred = self.model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        adult_recall = report['Adults (18+)']['recall']
        kids_precision = report['Kids & Family']['precision']
        
        self.assertGreaterEqual(
            adult_recall, 0.70,
            f"Recall của nhóm Adults (18+) ({adult_recall:.4f}) thấp hơn ngưỡng 0.70!"
        )
        self.assertGreaterEqual(
            kids_precision, 0.75,
            f"Precision của nhóm Kids & Family ({kids_precision:.4f}) thấp hơn ngưỡng 0.75!"
        )

    # =========================================================================
    # NHÓM 5: KIỂM TRA ĐỘ BỀN VỚI DỮ LIỆU BIÊN (ROBUSTNESS & EDGE CASES)
    # =========================================================================
    def test_13_edge_case_empty_and_spaces(self):
        """Kiểm tra mô hình không crash khi gặp chuỗi rỗng hoặc chỉ có khoảng trắng."""
        try:
            preds = self.model.predict(["", "   ", "\t\n"])
            self.assertEqual(len(preds), 3)
            for p in preds:
                self.assertIn(p, VALID_CLASSES)
        except Exception as e:
            self.fail(f"Mô hình bị văng lỗi khi nhận chuỗi rỗng: {e}")

    def test_14_edge_case_special_chars_and_numbers(self):
        """Kiểm tra mô hình không crash khi gặp chuỗi toàn số hoặc ký tự đặc biệt."""
        try:
            preds = self.model.predict(["12345 67890", "!@#$%^&*()_+", "??? ... !!!"])
            self.assertEqual(len(preds), 3)
            for p in preds:
                self.assertIn(p, VALID_CLASSES)
        except Exception as e:
            self.fail(f"Mô hình bị văng lỗi khi nhận ký tự đặc biệt: {e}")

    def test_15_determinism_consistency(self):
        """Kiểm tra tính nhất quán: Dự đoán 10 lần liên tiếp trên cùng đầu vào phải ra kết quả y hệt nhau."""
        test_input = ["Movie | Dramas | A dramatic story about family relationships."]
        first_pred = self.model.predict(test_input)[0]
        first_prob = self.model.predict_proba(test_input)[0]

        for _ in range(10):
            subsequent_pred = self.model.predict(test_input)[0]
            subsequent_prob = self.model.predict_proba(test_input)[0]
            self.assertEqual(first_pred, subsequent_pred)
            np.testing.assert_allclose(first_prob, subsequent_prob, rtol=1e-6)

# =============================================================================
# RUNNER: Trình thực thi bài test với thông báo chi tiết
# =============================================================================
def run_custom_test_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestNetflixModel)
    
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    print("\n" + "=" * 70)
    print("                        TỔNG KẾT KẾT QUẢ TEST")
    print("=" * 70)
    print(f"[*] Tổng số bài test đã chạy : {total}")
    print(f"[+] Bài test ĐẠT (PASS)       : {passed} / {total}")
    print(f"[-] Bài test THẤT BẠI (FAIL)  : {failures}")
    print(f"[!] Bài test BỊ LỖI (ERROR)   : {errors}")
    
    if result.wasSuccessful():
        print("\n>>> KẾT LUẬN: TẤT CẢ 15 BÀI TEST ĐÃ VƯỢT QUA XUẤT SẮC (100% PASS)!")
        print(">>> MÔ HÌNH ĐÃ ĐẠT CHUẨN ĐỘ CHÍNH XÁC, TÍNH TOÀN VẸN VÀ ĐỘ BỀN.")
    else:
        print("\n>>> KẾT LUẬN: CÓ BÀI TEST CHƯA ĐẠT, CẦN KIỂM TRA LẠI MÔ HÌNH.")
    print("=" * 70 + "\n")
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_custom_test_suite()
    sys.exit(0 if success else 1)
