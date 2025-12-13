import sys
import os
import argparse
from pathlib import Path
from mutagen import File as MutagenFile

# -----------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------

def hex_head(data, length=10):
    """显示二进制数据的前 N 个字节的十六进制表示"""
    if not data:
        return "Empty"
    hex_str = " ".join("{:02x}".format(c) for c in data[:length])
    return f"[{hex_str}...]"

def get_image_type(data):
    """根据文件头猜测图片格式"""
    if data.startswith(b'\xff\xd8\xff'):
        return "JPEG"
    elif data.startswith(b'\x89PNG'):
        return "PNG"
    elif data.startswith(b'GIF'):
        return "GIF"
    elif data.startswith(b'BM'):
        return "BMP"
    return "Unknown Binary"

def format_value(key, value, indent="    "):
    """
    智能格式化标签值。
    能够处理：普通字符串、列表、ID3帧对象、二进制图片数据。
    """
    val_type = type(value).__name__
    
    # 1. 检查是否有二进制数据 (图片/封面)
    # Mutagen 的图片通常在 data 属性中，或者本身就是 bytes
    binary_data = None
    if hasattr(value, 'data') and isinstance(value.data, bytes):
        binary_data = value.data
    elif isinstance(value, bytes):
        # 有些老旧标签可能是纯 bytes，如果不长，视为文本，太长视为二进制
        if len(value) > 256 or b'\0' in value[:10]: 
            binary_data = value

    if binary_data:
        size_kb = len(binary_data) / 1024
        img_fmt = get_image_type(binary_data)
        hex_preview = hex_head(binary_data)
        mime = getattr(value, 'mime', 'N/A')
        desc = getattr(value, 'desc', 'N/A')
        type_id = getattr(value, 'type', 'N/A') # ID3 APIC type (3=Cover Front)
        
        return (f"\n{indent}📦 [BINARY/IMAGE DETECTED]\n"
                f"{indent}   Format : {img_fmt}\n"
                f"{indent}   Size   : {size_kb:.2f} KB\n"
                f"{indent}   MIME   : {mime}\n"
                f"{indent}   Desc   : {desc}\n"
                f"{indent}   PicType: {type_id}\n"
                f"{indent}   Header : {hex_preview}")

    # 2. 处理 ID3 文本帧 (通常包含 text 属性，且是列表)
    if hasattr(value, 'text'):
        # ID3 timestamp objects 等特殊处理
        return f"{value.text} (ID3 Frame)"

    # 3. 处理列表 (FLAC/Vorbis comments 经常是列表)
    if isinstance(value, list):
        return f"{value} (List len={len(value)})"

    # 4. 默认转字符串
    return str(value)

# -----------------------------------------------------------
# 主逻辑
# -----------------------------------------------------------

def inspect_file(file_path):
    path = Path(file_path)
    print("="*60)
    print(f"📂 分析文件: {path.name}")
    print(f"📍 完整路径: {path.absolute()}")
    
    if not path.exists():
        print("❌ 文件不存在")
        return

    try:
        # 使用 Mutagen 通用加载
        audio = MutagenFile(path)
        
        if not audio:
            print("❌ Mutagen 无法识别此文件格式 (或非音频文件)")
            return
            
        print(f"🧩 Mutagen 对象类型: {type(audio)}")

        # --- 第一部分：流信息 (Stream Info) ---
        print("\n" + "-"*20 + " [音频流信息] " + "-"*20)
        if audio.info:
            # 动态遍历 info 对象的所有属性
            info_attrs = [attr for attr in dir(audio.info) if not attr.startswith("_") and not callable(getattr(audio.info, attr))]
            for attr in info_attrs:
                val = getattr(audio.info, attr)
                # 过滤掉太长的调试信息
                if isinstance(val, (str, bytes)) and len(val) > 50:
                    val = f"{str(val)[:50]}..."
                print(f"{attr:<15}: {val}")
            
            # 专门打印直观的时长
            if hasattr(audio.info, 'length'):
                m, s = divmod(audio.info.length, 60)
                print(f"{'Duration':<15}: {int(m)}m {int(s)}s")
        else:
            print("   (无流信息)")

        # --- 第二部分：元数据标签 (Tags) ---
        print("\n" + "-"*20 + " [元数据标签] " + "-"*20)
        
        if not audio.tags:
            print("   (无标签数据)")
        else:
            print(f"🏷️  Tags 类型: {type(audio.tags)}")
            count = 0
            
            # 获取所有 Keys。有些格式是 dict，有些是类 dict
            keys = audio.tags.keys()
            
            for key in keys:
                count += 1
                val = audio.tags[key]
                formatted_val = format_value(key, val)
                print(f"🔹 [{key}] : {formatted_val}")
            
            print(f"\n✅ 共扫描到 {count} 个标签项")

    except Exception as e:
        print(f"❌ 读取错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="万能音频元数据查看器 (基于 Mutagen)")
    parser.add_argument("files", nargs='+', help="要检查的一个或多个音频文件路径")
    
    args = parser.parse_args()

    for f in args.files:
        inspect_file(f)
        print("\n")

if __name__ == "__main__":
    main()