"""T5.3 上传清单：本机→云端（打包一次、scp 上去、bash setup.sh 即可）。

打包：cd /Users/yangkun/Desktop/Projects/onwer/Qwen_JEY_rag_judge && bash train/laya/pack.sh
上传：scp laya_ft.tar.gz root@<IP>:/root/ && ssh root@<IP> "cd /root && tar xzf laya_ft.tar.gz && bash laya_ft/setup.sh"
"""
