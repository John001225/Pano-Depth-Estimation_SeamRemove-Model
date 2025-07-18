import subprocess
from pathlib import Path
from datetime import datetime

# 設定要執行的 Python 檔案路徑
scripts = [
    "/home/user/sheng/panoDepth/AdelaiDepth/LeReS/Minist_Test/tools/test_depth.py",
    "/home/user/sheng/panoDepth/Depth-Anything-V2/metric_depth/run.py",
    "/home/user/sheng/panoDepth/Depth-Anything/metric_depth/depth_pred.py",
    "/home/user/sheng/panoDepth/ZoeDepth/ZoeDepth_saveRawImg.py"
]

for script_path in scripts:
    script = Path(script_path)
    log_file = f"{script.stem}_fold565_execution_log_2.txt"

    with open(log_file, "w") as log:
        print(f"\nExecuting {script.name} in {script.parent}...\n{'-'*40}")
        log.write(f"Executing {script.name} in {script.parent}\n{'-'*40}\n")

        start_time = datetime.now()
        
        # 使用 subprocess.run() 執行腳本
        result = subprocess.run(
            f"python {script}",  # 使用 python 執行檔案
            cwd=script.parent,   # 設定工作目錄為檔案所在資料夾
            shell=True           # 在終端中執行
        )
        
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        
        # 檢查返回碼以確定是否成功
        if result.returncode == 0:
            print(f"{'-'*40}\nSuccessfully executed {script.name} in {elapsed_time}\n")
            log.write(f"{'-'*40}\nSuccessfully executed {script.name} in {elapsed_time}\n\n")
        else:
            print(f"Error executing {script.name} (exit code {result.returncode})\n")
            log.write(f"Error executing {script.name} (exit code {result.returncode})\n\n")
            break  # 如果失敗則停止執行

# import subprocess
# from pathlib import Path
# from datetime import datetime

# # 設定要執行的 Python 檔案路徑
# scripts = [
#     "/home/user/sheng/panoDepth/AdelaiDepth/LeReS/Minist_Test/tools/test_depth.py",
#     "/home/user/sheng/panoDepth/Depth-Anything-V2/metric_depth/run.py",
#     "/home/user/sheng/panoDepth/Depth-Anything/metric_depth/depth_pred.py",
#     "/home/user/sheng/panoDepth/ZoeDepth/ZoeDepth_saveRawImg.py"
# ]

# # 記錄檔案名稱
# log_file = "execution_log_565.txt"

# with open(log_file, "w") as log:
#     for script_path in scripts:
#         script = Path(script_path)
#         print(f"\nExecuting {script.name} in {script.parent}...\n{'-'*40}")
#         log.write(f"Executing {script.name} in {script.parent}\n{'-'*40}\n")

#         start_time = datetime.now()
        
#         # 使用 subprocess.run() 執行腳本
#         result = subprocess.run(
#             f"python {script}",  # 使用 python 執行檔案
#             cwd=script.parent,   # 設定工作目錄為檔案所在資料夾
#             shell=True           # 在終端中執行
#         )
        
#         end_time = datetime.now()
#         elapsed_time = end_time - start_time
        
#         # 檢查返回碼以確定是否成功
#         if result.returncode == 0:
#             print(f"{'-'*40}\nSuccessfully executed {script.name} in {elapsed_time}\n")
#             log.write(f"{'-'*40}\nSuccessfully executed {script.name} in {elapsed_time}\n\n")
#         else:
#             print(f"Error executing {script.name} (exit code {result.returncode})\n")
#             log.write(f"Error executing {script.name} (exit code {result.returncode})\n\n")
#             break  # 如果失敗則停止執行

# import subprocess
# from pathlib import Path

# # 設定要執行的 Python 檔案路徑
# scripts = [
#     "/home/user/sheng/panoDepth/AdelaiDepth/LeReS/Minist_Test/tools/test_depth.py",
#     "/home/user/sheng/panoDepth/Depth-Anything-V2/metric_depth/run.py",
#     "/home/user/sheng/panoDepth/Depth-Anything/metric_depth/depth_pred.py",
#     "/home/user/sheng/panoDepth/ZoeDepth/ZoeDepth_saveRawImg.py"
# ]

# # 按順序在終端中執行每個 Python 檔案
# for script_path in scripts:
#     script = Path(script_path)
#     print(f"\nExecuting {script.name} in {script.parent}...\n{'-'*40}")
    
#     # 使用 `subprocess.run()` 在終端中執行每個腳本
#     result = subprocess.run(
#         f"python {script}",  # 使用 `python` 執行檔案
#         cwd=script.parent,   # 設定工作目錄為檔案所在資料夾
#         shell=True           # 在終端中執行
#     )

#     # 檢查返回碼以確定是否成功
#     if result.returncode == 0:
#         print(f"{'-'*40}\nSuccessfully executed {script.name}\n")
#     else:
#         print(f"Error executing {script.name} (exit code {result.returncode})\n")
#         break  # 如果失敗則停止執行
