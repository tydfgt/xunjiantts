# 树莓派 eSpeak 中文语音合成配置文档

> **项目**: 树莓派 TTS 语音系统  
> **日期**: 2026-06-08  
> **设备**: Raspberry Pi 5 Model B (aarch64)  
> **系统**: Debian (Raspberry Pi OS)  

---

## 一、环境概览

| 项目 | 详情 |
|------|------|
| 硬件 | Raspberry Pi 5 Model B Rev 1.1 |
| 架构 | aarch64 (64位 ARM) |
| Python 环境 | Miniconda3 (conda 26.1.1) |
| 虚拟环境 | `tts` (Python 3.11) |
| 虚拟环境路径 | `/home/cedarq/miniconda3/envs/tts/` |

---

## 二、安装过程

### 2.1 安装 eSpeak / eSpeak-NG

```bash
# 安装经典 eSpeak（已包含中文基础支持）
sudo apt install -y espeak

# 安装新一代 eSpeak-NG（中文支持更好）
sudo apt install -y espeak-ng espeak-ng-data
```

### 2.2 安装 MBROLA 中文女声语音包

MBROLA 是第三方高质量语音合成引擎，eSpeak-NG 可调用 MBROLA 的语音数据来获得更自然的发音。

```bash
# 安装 MBROLA 引擎 + 中文女声语音数据
sudo apt install -y mbrola mbrola-cn1
```

> **语音包说明**: `mbrola-cn1` 是中文女声语音数据，安装后 eSpeak-NG 可通过 `-v zh` 参数调用。

### 2.3 验证安装

```bash
# 查看版本
espeak-ng --version

# 查看所有中文可用语音
espeak-ng --voices=zh
```

预期输出：

```
Pty Language       Age/Gender VoiceName                          File
 5  cmn             --/M      Chinese_(Mandarin)                  sit/cmn
 5  cmn-latn-pinyin --/M      Chinese_(Mandarin,_latin_as_Pinyin) sit/cmn-Latn-pinyin
 5  zh              --/F      chinese-mb-cn1                      mb/mb-cn1
 5  yue             --/M      Chinese_(Cantonese)                 sit/yue
```

---

## 三、中文语音选项

| 语音参数 | 说明 | 音色 | 推荐场景 |
|----------|------|------|----------|
| `-v zh` | MBROLA 中文女声 | 👩 女声 | **推荐**，最自然的中文发音 |
| `-v cmn` | eSpeak-NG 自带普通话 | 👨 男声 | 备用方案，无需额外依赖 |
| `-v yue` | 粤语 | 👨 男声 | 粤语播报 |

---

## 四、常用命令

### 4.1 基本语法

```bash
espeak-ng -v <语音> -s <语速> -a <音量> -p <音调> "文本内容"
```

### 4.2 参数说明

| 参数 | 说明 | 取值范围 | 默认值 |
|------|------|----------|--------|
| `-v` | 语音/语言 | `zh`, `cmn`, `en` 等 | `en` |
| `-s` | 语速 (字/分钟) | 80 ~ 400 | 175 |
| `-a` | 音量 | 0 ~ 200 | 100 |
| `-p` | 音调 | 0 ~ 99 | 50 |
| `-w` | 输出为 WAV 文件 | 文件路径 | - |
| `-f` | 从文件读取文本 | 文件路径 | - |

### 4.3 使用示例

```bash
# 中文女声（MBROLA，推荐）
espeak-ng -v zh -s 130 "你好世界，欢迎使用树莓派语音系统"

# 中文男声（eSpeak-NG 自带）
espeak-ng -v cmn -s 130 "今天天气不错"

# 调整语速和音调
espeak-ng -v zh -s 100 -p 60 "慢速低音效果"

# 输出到 WAV 文件
espeak-ng -v zh -s 130 -w output.wav "把语音保存为文件"

# 从文本文件读取
espeak-ng -v zh -s 130 -f /path/to/text.txt

# 英文语音（带变体）
espeak-ng -v en+f3 -s 150 "Hello, this is a test."    # 英文女声
espeak-ng -v en-us+m1 -s 150 "Hello from Raspberry Pi" # 美式男声
```

---

## 五、在 Conda 虚拟环境中使用

### 5.1 激活环境

```bash
source ~/miniconda3/bin/activate
conda activate tts
```

### 5.2 确认 espeak-ng 可用

```bash
# espeak-ng 是系统级安装，conda 环境中可直接调用
which espeak-ng
# 输出: /usr/bin/espeak-ng
```

### 5.3 Python 调用方式

```python
import os
import subprocess

# 方式一：os.system（简单直接）
def speak_cn(text, speed=130):
    """中文语音合成"""
    os.system(f'espeak-ng -v zh -s {speed} "{text}"')

# 方式二：subprocess（更安全，避免 shell 注入）
def speak_cn_safe(text, speed=130):
    """中文语音合成（安全版本）"""
    subprocess.run(
        ['espeak-ng', '-v', 'zh', '-s', str(speed), text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# 方式三：生成 WAV 文件
def text_to_wav(text, output_path, speed=130):
    """将文本转为 WAV 音频文件"""
    subprocess.run(
        ['espeak-ng', '-v', 'zh', '-s', str(speed), '-w', output_path, text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return output_path

# 使用示例
if __name__ == '__main__':
    speak_cn("你好，语音系统已就绪")
    text_to_wav("这是保存为文件的语音", "/tmp/test.wav")
```

---

## 六、音频输出配置

### 6.1 检查音频设备

```bash
# 查看音频设备
aplay -l
# 或
cat /proc/asound/cards
```

### 6.2 常见问题

| 问题 | 解决方案 |
|------|----------|
| 无声音输出 | 检查 HDMI/3.5mm 音频输出设置: `sudo raspi-config` → System → Audio |
| 权限错误 | `sudo usermod -aG audio $USER` 然后重新登录 |
| 中文乱码 | 确保终端使用 UTF-8 编码: `echo $LANG` |

---

## 七、依赖关系图

```
eSpeak-NG (语音合成引擎)
  ├── espeak-ng-data (基础语音数据)
  │     ├── cmn (普通话男声)
  │     ├── yue (粤语)
  │     └── en, ja, fr ... (其他语言)
  └── MBROLA (高质量语音引擎)
        └── mbrola-cn1 (中文女声数据)
              └── 通过 -v zh 调用
```

---

## 八、与 Vosk 配合使用

eSpeak-NG 负责 TTS（文字→语音），Vosk 负责 ASR（语音→文字），两者配合可实现完整的语音交互系统：

```python
# 完整语音交互示例
import os
import subprocess

def tts(text, speed=130):
    """文字转语音"""
    subprocess.run(['espeak-ng', '-v', 'zh', '-s', str(speed), text],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 使用
tts("你好，我是树莓派语音助手")
```

> **注意**: eSpeak-NG 合成的机械音无法被 Vosk 准确识别（ASR 模型基于真实人声训练）。如需录制真实语音进行识别测试，请使用麦克风。

---

## 九、卸载方法

```bash
# 卸载 eSpeak
sudo apt remove --purge espeak espeak-ng espeak-ng-data

# 卸载 MBROLA
sudo apt remove --purge mbrola mbrola-cn1

# 清理无用依赖
sudo apt autoremove --purge
```

---

## 十、参考资源

- eSpeak-NG 官方: https://github.com/espeak-ng/espeak-ng
- MBROLA 项目: https://github.com/numediart/MBROLA
- 百度开发者文章: https://developer.baidu.com/article/detail.html?id=3896188
- CSDN 语音变体指南: https://ask.csdn.net/questions/8849780
