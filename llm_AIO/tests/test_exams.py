"""
考试模块 MVP 集成测试（requests）。

覆盖：题库（单选/多选/判断）→ 创建考试 → 发布 → 学生开始/提交 → 自动阅卷 → 学生查成绩。
"""

import requests
import json
import sys
import os
from datetime import datetime, timedelta, timezone


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "http://localhost:8000/api/v1"

access_token = None


def print_result(test_name, success, message="", data=None):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} | {test_name}")
    if message:
        print(f"      {message}")
    if data is not None:
        try:
            print(f"      响应: {json.dumps(data, ensure_ascii=False, indent=2)[:1200]}")
        except Exception:
            pass
    print()


def login(username, password="admin123"):
    r = requests.post(
        f"{BASE_URL}/auth/login",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=10,
    )
    return r


def ensure_user_as_admin(admin_token, role, username, email, password="admin123", student_id=None, teacher_id=None):
    # 先登录（存在则复用）
    lr = login(username, password=password)
    if lr.status_code == 200:
        return lr.json()["access_token"], lr.json()["user_id"]

    r = requests.post(
        f"{BASE_URL}/users/",
        headers={"Authorization": f"Bearer {admin_token}", "Content-Type": "application/json"},
        json={
            "username": username,
            "email": email,
            "full_name": username,
            "password": password,
            "role": role,
            "student_id": student_id,
            "teacher_id": teacher_id,
        },
        timeout=10,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(r.text)
    lr2 = login(username, password=password)
    if lr2.status_code != 200:
        raise RuntimeError(lr2.text)
    return lr2.json()["access_token"], lr2.json()["user_id"]


def run():
    print("=" * 60)
    print("考试模块 MVP 集成测试")
    print("=" * 60)
    print()

    # admin
    a = login("admin", "admin123")
    if a.status_code != 200:
        print_result("管理员登录", False, f"状态码: {a.status_code}", a.text)
        sys.exit(1)
    admin_token = a.json()["access_token"]
    print_result("管理员登录", True, "登录成功", {"role": a.json().get("role")})

    # teacher + student
    teacher_token, _ = ensure_user_as_admin(admin_token, "teacher", "exam_teacher", "exam_teacher@example.com", teacher_id="exam_teacher")
    student_token, _ = ensure_user_as_admin(admin_token, "student", "exam_student", "exam_student@example.com", student_id="exam_student")

    # ---------- 权限边界 ----------
    denied_create_q = requests.post(
        f"{BASE_URL}/exams/question-bank",
        headers={"Authorization": f"Bearer {student_token}", "Content-Type": "application/json"},
        json={
            "question_type": "single",
            "stem": "forbidden",
            "options": [{"key": "A", "text": "x"}],
            "correct_answer": "A",
            "score": 1.0,
        },
        timeout=10,
    )
    if denied_create_q.status_code == 403:
        print_result("权限边界：学生创建题目", True, "返回 403（符合预期）", denied_create_q.json())
    else:
        print_result("权限边界：学生创建题目", False, f"期望 403，实际 {denied_create_q.status_code}", denied_create_q.text)
        sys.exit(1)

    # 1) 创建题目（单选/多选/判断）
    def create_q(payload):
        r = requests.post(
            f"{BASE_URL}/exams/question-bank",
            headers={"Authorization": f"Bearer {teacher_token}", "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        return r

    q1 = create_q({
        "question_type": "single",
        "stem": "1+1=?",
        "options": [{"key": "A", "text": "1"}, {"key": "B", "text": "2"}],
        "correct_answer": "B",
        "score": 2.0
    })
    q2 = create_q({
        "question_type": "multiple",
        "stem": "选择质数",
        "options": [{"key": "A", "text": "2"}, {"key": "B", "text": "3"}, {"key": "C", "text": "4"}],
        "correct_answer": ["A", "B"],
        "score": 3.0
    })
    q3 = create_q({
        "question_type": "true_false",
        "stem": "地球是圆的",
        "options": None,
        "correct_answer": "true",
        "score": 1.0
    })
    if any(r.status_code not in (200, 201) for r in (q1, q2, q3)):
        print_result("创建题目", False, "创建失败", {"q1": q1.text, "q2": q2.text, "q3": q3.text})
        sys.exit(1)
    qids = [q1.json()["id"], q2.json()["id"], q3.json()["id"]]
    print_result("创建题目", True, f"qids={qids}", qids)

    # 2) 创建考试并发布
    exam = requests.post(
        f"{BASE_URL}/exams/",
        headers={"Authorization": f"Bearer {teacher_token}", "Content-Type": "application/json"},
        json={
            "competition_id": None,
            "title": "MVP Exam",
            "description": "desc",
            "duration_minutes": 30,
            "question_ids": qids,
        },
        timeout=10,
    )
    if exam.status_code not in (200, 201):
        print_result("创建考试", False, f"状态码: {exam.status_code}", exam.text)
        sys.exit(1)
    exam_id = exam.json()["id"]

    pub = requests.put(
        f"{BASE_URL}/exams/{exam_id}/publish",
        headers={"Authorization": f"Bearer {teacher_token}"},
        timeout=10,
    )
    if pub.status_code != 200:
        print_result("发布考试", False, f"状态码: {pub.status_code}", pub.text)
        sys.exit(1)
    print_result("发布考试", True, "成功", pub.json())

    # 时间窗校验：创建一个已结束的考试，学生 start 应返回 400
    ended_exam = requests.post(
        f"{BASE_URL}/exams/",
        headers={"Authorization": f"Bearer {teacher_token}", "Content-Type": "application/json"},
        json={
            "competition_id": None,
            "title": "Ended Exam",
            "description": "ended",
            "start_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "end_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "duration_minutes": 10,
            "question_ids": qids,
        },
        timeout=10,
    )
    if ended_exam.status_code not in (200, 201):
        print_result("创建已结束考试", False, f"状态码: {ended_exam.status_code}", ended_exam.text)
        sys.exit(1)
    ended_exam_id = ended_exam.json()["id"]
    pub2 = requests.put(
        f"{BASE_URL}/exams/{ended_exam_id}/publish",
        headers={"Authorization": f"Bearer {teacher_token}"},
        timeout=10,
    )
    if pub2.status_code != 200:
        print_result("发布已结束考试", False, f"状态码: {pub2.status_code}", pub2.text)
        sys.exit(1)
    denied_start = requests.post(
        f"{BASE_URL}/exams/{ended_exam_id}/start",
        headers={"Authorization": f"Bearer {student_token}"},
        timeout=10,
    )
    if denied_start.status_code == 400:
        print_result("时间窗：已结束考试不可开始", True, "返回 400（符合预期）", denied_start.json())
    else:
        print_result("时间窗：已结束考试不可开始", False, f"期望 400，实际 {denied_start.status_code}", denied_start.text)
        sys.exit(1)

    # 3) 学生开始考试并提交答案（含对错）
    start = requests.post(
        f"{BASE_URL}/exams/{exam_id}/start",
        headers={"Authorization": f"Bearer {student_token}"},
        timeout=10,
    )
    if start.status_code != 200:
        print_result("开始考试", False, f"状态码: {start.status_code}", start.text)
        sys.exit(1)

    submit = requests.post(
        f"{BASE_URL}/exams/{exam_id}/submit",
        headers={"Authorization": f"Bearer {student_token}", "Content-Type": "application/json"},
        json={
            "answers": [
                {"question_id": qids[0], "answer": "B"},          # 对
                {"question_id": qids[1], "answer": ["A", "B"]},   # 对
                {"question_id": qids[2], "answer": "false"},      # 错
            ]
        },
        timeout=10,
    )
    if submit.status_code != 200:
        print_result("提交考试", False, f"状态码: {submit.status_code}", submit.text)
        sys.exit(1)
    total = submit.json().get("total_score")
    # 2 + 3 + 0 = 5
    if total != 5.0:
        print_result("自动阅卷总分", False, f"期望 5.0，实际 {total}", submit.json())
        sys.exit(1)
    print_result("提交考试 + 自动阅卷", True, f"total_score={total}", submit.json())

    # 4) 学生查询成绩
    me = requests.get(
        f"{BASE_URL}/exams/{exam_id}/attempts/me",
        headers={"Authorization": f"Bearer {student_token}"},
        timeout=10,
    )
    if me.status_code != 200:
        print_result("学生查成绩", False, f"状态码: {me.status_code}", me.text)
        sys.exit(1)
    print_result("学生查成绩", True, "成功", me.json())

    # 权限边界：学生不能查看监考视角 attempt 列表
    denied_attempts = requests.get(
        f"{BASE_URL}/exams/{exam_id}/attempts",
        headers={"Authorization": f"Bearer {student_token}"},
        timeout=10,
    )
    if denied_attempts.status_code == 403:
        print_result("权限边界：学生查看attempt列表", True, "返回 403（符合预期）", denied_attempts.json())
    else:
        print_result("权限边界：学生查看attempt列表", False, f"期望 403，实际 {denied_attempts.status_code}", denied_attempts.text)
        sys.exit(1)

    # 监考/教师查看 attempt 列表与详情
    attempts = requests.get(
        f"{BASE_URL}/exams/{exam_id}/attempts",
        headers={"Authorization": f"Bearer {teacher_token}"},
        timeout=10,
    )
    if attempts.status_code != 200 or not attempts.json():
        print_result("监考：教师查看attempt列表", False, f"状态码: {attempts.status_code}", attempts.text)
        sys.exit(1)
    attempt_id = attempts.json()[0]["id"]
    print_result("监考：教师查看attempt列表", True, f"attempt_id={attempt_id}", attempts.json()[:2])

    attempt_detail = requests.get(
        f"{BASE_URL}/exams/attempts/{attempt_id}",
        headers={"Authorization": f"Bearer {teacher_token}"},
        timeout=10,
    )
    if attempt_detail.status_code != 200:
        print_result("监考：教师查看attempt详情", False, f"状态码: {attempt_detail.status_code}", attempt_detail.text)
        sys.exit(1)
    if not attempt_detail.json().get("answers"):
        print_result("监考：attempt答案明细", False, "answers 为空", attempt_detail.json())
        sys.exit(1)
    print_result("监考：教师查看attempt详情", True, "成功", {"total_score": attempt_detail.json().get("total_score"), "answers_count": len(attempt_detail.json().get("answers", []))})

    # 监考强制交卷：对一个新 attempt（只 start 不 submit）执行 force-submit，应返回 200 且 total_score 存在
    exam2 = requests.post(
        f"{BASE_URL}/exams/",
        headers={"Authorization": f"Bearer {teacher_token}", "Content-Type": "application/json"},
        json={
            "competition_id": None,
            "title": "Force Submit Exam",
            "description": "force",
            "duration_minutes": 30,
            "question_ids": qids,
        },
        timeout=10,
    )
    if exam2.status_code not in (200, 201):
        print_result("创建强制交卷考试", False, f"状态码: {exam2.status_code}", exam2.text)
        sys.exit(1)
    exam2_id = exam2.json()["id"]
    pub3 = requests.put(
        f"{BASE_URL}/exams/{exam2_id}/publish",
        headers={"Authorization": f"Bearer {teacher_token}"},
        timeout=10,
    )
    if pub3.status_code != 200:
        print_result("发布强制交卷考试", False, f"状态码: {pub3.status_code}", pub3.text)
        sys.exit(1)

    start2 = requests.post(
        f"{BASE_URL}/exams/{exam2_id}/start",
        headers={"Authorization": f"Bearer {student_token}"},
        timeout=10,
    )
    if start2.status_code != 200:
        print_result("开始强制交卷考试", False, f"状态码: {start2.status_code}", start2.text)
        sys.exit(1)

    attempts2 = requests.get(
        f"{BASE_URL}/exams/{exam2_id}/attempts",
        headers={"Authorization": f"Bearer {teacher_token}"},
        timeout=10,
    )
    if attempts2.status_code != 200 or not attempts2.json():
        print_result("监考：获取attempt列表(强制交卷考试)", False, f"状态码: {attempts2.status_code}", attempts2.text)
        sys.exit(1)
    attempt2_id = attempts2.json()[0]["id"]

    forced = requests.post(
        f"{BASE_URL}/exams/attempts/{attempt2_id}/force-submit",
        headers={"Authorization": f"Bearer {teacher_token}"},
        timeout=10,
    )
    if forced.status_code != 200:
        print_result("监考：强制交卷", False, f"状态码: {forced.status_code}", forced.text)
        sys.exit(1)
    if forced.json().get("status") != "graded":
        print_result("监考：强制交卷状态", False, "status 不是 graded", forced.json())
        sys.exit(1)
    print_result("监考：强制交卷", True, "成功", {"attempt_id": attempt2_id, "total_score": forced.json().get("total_score")})

    print_result("考试模块测试完成", True, "MVP 用例通过")


if __name__ == "__main__":
    run()

