from apscheduler.schedulers.background import BackgroundScheduler
import time

def test_job():
    print("测试任务运行成功！")

scheduler = BackgroundScheduler()
scheduler.add_job(test_job, 'interval', seconds=3)
scheduler.start()

print("调度器已启动")
try:
    # 运行10秒后退出
    time.sleep(10)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown() 