# mangda-detection
เป็นโปคเจคแยกประเภท และระบุตำแหน่งของแมงดาทะเลได้
* [mediumโปรเจค](https://medium.com/@phakkhaphonartburai/ai-%E0%B9%81%E0%B8%A2%E0%B8%81%E0%B9%81%E0%B8%A2%E0%B8%B0%E0%B9%81%E0%B8%A1%E0%B8%87%E0%B8%94%E0%B8%B2%E0%B8%88%E0%B8%B2%E0%B8%99-%E0%B8%81%E0%B8%B1%E0%B8%9A-%E0%B9%81%E0%B8%A1%E0%B8%87%E0%B8%94%E0%B8%B2%E0%B8%9E%E0%B8%B4%E0%B8%A9-784bf470c592)
## แนะนำการใช้งานเบื้องต้น
### เนื่องจาก library icevisionไม่supportบนwindows จึงไม่สามารถใช้บน windows ได้
* โหลดไฟล์โมเดลเพิ่มเติมเนื่องจากข้อจำกัดของ github   
  * โหลดโมเดลทั้งหมด
  * https://drive.google.com/drive/folders/1oceon16uvdo5fn4XAJUEyYtXqHmza6jk?usp=sharing
  * นำโฟลเดอร์โมเดลใส่ใน mangda-detection
* วิธีการติดตั้ง libraries เพิ่มเติมเนื่องจากไม่สามารถลงใน requirements.txt ได้
  * pip3 install -r requirements.txt
  * pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
  * pip3 install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
