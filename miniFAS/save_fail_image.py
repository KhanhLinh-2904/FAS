import glob
import os
from PIL import Image


with open('/home/linhhima/FAS_Vinh/FAS_CNN/log.txt', 'r') as file:
    for line in file:
        if not os.path.exists('/home/linhhima/FAS_Vinh/FAS_CNN/db_fake_fail'):
            os.mkdir('/home/linhhima/FAS_Vinh/FAS_CNN/db_fake_fail')
        if not os.path.exists('/home/linhhima/FAS_Vinh/FAS_CNN/db_real_fail'):
            os.mkdir('/home/linhhima/FAS_Vinh/FAS_CNN/db_real_fail')
        if "fake/" in line:
            try:
                # Mở ảnh từ đường dẫn
                img_path = line.strip()
                with Image.open(img_path) as img:
                    # Tách tên file từ đường dẫn
                    file_name = os.path.basename(img_path)
                    # Tạo đường dẫn mới để lưu ảnh
                    save_path = os.path.join('/home/linhhima/FAS_Vinh/FAS_CNN/db_fake_fail', file_name)
                    # Lưu ảnh vào thư mục đích
                    img.save(save_path)
            except Exception as e:
                print(f"Lỗi: {e}")
        
        if "real/" in line:
            try:
                # Mở ảnh từ đường dẫn
                img_path = line.strip()
                with Image.open(img_path) as img:
                    # Tách tên file từ đường dẫn
                    file_name = os.path.basename(img_path)
                    # Tạo đường dẫn mới để lưu ảnh
                    save_path = os.path.join('/home/linhhima/FAS_Vinh/FAS_CNN/db_real_fail', file_name)
                    # Lưu ảnh vào thư mục đích
                    img.save(save_path)
            except Exception as e:
                print(f"Lỗi: {e}")