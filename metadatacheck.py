import os
from pathlib import Path
from mutagen import File as MutagenFile
from mutagen.id3 import ID3, APIC

# --- 你的路径 ---
MUSIC_DIR = Path("J:\音乐\Hitorie")
# ----------------

def hex_head(data):
    """显示前10个字节的十六进制，用于判断文件头 (如 FF D8 FF 是 JPEG)"""
    return " ".join("{:02x}".format(c) for c in data[:10])

def scan_covers():
    print(f"🔍 正在扫描: {MUSIC_DIR} ...\n")
    
    files = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.flac"))
    
    # 找到那个特定的 Hitorie 文件进行测试，或者读取第一个文件
    target_file = None
    for f in files:
        if "5カウントハロー" in f.name:
            target_file = f
            break
    
    if not target_file and files:
        target_file = files[0]
    
    if not target_file:
        print("❌ 未找到音乐文件")
        return

    print(f"📂 目标文件: {target_file.name}")
    
    try:
        # 方法 A: 通用 File 读取
        f = MutagenFile(target_file)
        print(f"   对象类型: {type(f)}")
        
        if f.tags:
            print(f"   Tags 类型: {type(f.tags)}")
            
            # 1. 遍历所有 Key，寻找含有 'APIC' 或 'Picture' 字样的
            print("\n   --- [1] 遍历 Key 查找 ---")
            found_in_keys = False
            for key in f.tags.keys():
                key_str = str(key)
                val = f.tags[key]
                if "APIC" in key_str or "PIC" in key_str:
                    found_in_keys = True
                    print(f"   ✅ 发现疑似封面 Key: '{key_str}'")
                    print(f"      类型: {type(val)}")
                    if hasattr(val, 'data'):
                        print(f"      包含 .data 属性! 大小: {len(val.data)} bytes")
                        print(f"      文件头: {hex_head(val.data)}")
                    else:
                        print("      ❌ 无 .data 属性")
            if not found_in_keys:
                print("   ❌ 未在 Keys 中找到 'APIC' 字样")

            # 2. 暴力扫描所有值，寻找二进制数据
            print("\n   --- [2] 暴力扫描值 (寻找大块二进制) ---")
            for key, val in f.tags.items():
                # 检查 .data 属性
                binary_data = None
                if hasattr(val, 'data'):
                    binary_data = val.data
                elif isinstance(val, bytes):
                    binary_data = val
                
                if binary_data and len(binary_data) > 1000: # 大于 1KB 可能是图片
                    print(f"   ✅ Key: '{key}' 包含 {len(binary_data)} 字节数据")
                    print(f"      类型: {type(val)}")
                    print(f"      文件头: {hex_head(binary_data)}")
                    if b'\xff\xd8\xff' in binary_data[:10]:
                        print("      👉 这是一个 JPEG 图片!")
                    elif b'\x89PNG' in binary_data[:10]:
                        print("      👉 这是一个 PNG 图片!")
                    else:
                        print("      ❓ 未知格式")

        # 方法 B: 强制作为 ID3 读取 (针对 MP3)
        if target_file.suffix.lower() == ".mp3":
            print("\n   --- [3] 强制 ID3 模式读取 ---")
            try:
                audio = ID3(target_file)
                # 使用 ID3 专用的 getall 方法
                apic_frames = audio.getall("APIC")
                print(f"   audio.getall('APIC') 返回了 {len(apic_frames)} 个帧")
                if apic_frames:
                    first = apic_frames[0]
                    print(f"   第一帧数据大小: {len(first.data)} bytes")
                    print(f"   MIME: {first.mime}")
            except Exception as e:
                print(f"   ID3 读取失败: {e}")

    except Exception as e:
        print(f"❌ 严重错误: {e}")

if __name__ == "__main__":
    scan_covers()