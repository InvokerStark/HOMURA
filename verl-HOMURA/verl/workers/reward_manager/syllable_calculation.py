import pyphen
import re
import fugashi

def cal_syllable_count(text, lang='en'):
    """
    精确计算多种语言的音节数量
    
    参数:
    text: 要分析的文本字符串
    lang: 语言代码 ('zh'-中文, 'en'-英文, 'de'-德文, 'fr'-法文)
    
    返回:
    音节数量
    """
    if not text or not text.strip():
        return 0
        
    # 清理文本：去除多余空格和标点
    cleaned_text = re.sub(r'[^\w\s]', ' ', text.strip())
    words = [word for word in cleaned_text.split() if word]
    
    if not words:
        return 0
        
    total_syllables = 0
    
    # 中文音节计数（每个汉字通常算一个音节）
    if lang.lower() == 'zh':
        # 统计中文字符（Unicode中的汉字范围）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', cleaned_text)
        total_syllables = len(chinese_chars)
        
        # 处理可能混入的非中文单词
        non_chinese_words = re.findall(r'[a-zA-Z]+', cleaned_text)
        for word in non_chinese_words:
            total_syllables += _pyphen_syllable_count(word, 'en')
    
    # 日语音节计数
    elif lang.lower() == 'ja':
        total_syllables = _japanese_syllable_count(cleaned_text)
    
    # 其他语言使用Pyphen
    else:
        lang_map = {
            'en': 'en_US',  # 英语
            'de': 'de_DE',  # 德语
            'fr': 'fr_FR',   # 法语
            'es': 'es_ES'   # 新增：西班牙语 [6](@ref)
        }
        
        pyphen_lang = lang_map.get(lang.lower(), 'en_US')
        
        for word in words:
            # 跳过纯数字
            if word.isdigit():
                total_syllables += 1
                continue
                
            syllable_count = _pyphen_syllable_count(word, pyphen_lang)
            total_syllables += syllable_count
    
    return total_syllables

def _pyphen_syllable_count(word, pyphen_lang):
    """使用Pyphen计算单个单词的音节数"""
    try:
        dic = pyphen.Pyphen(lang=pyphen_lang)
        hyphenated = dic.inserted(word)
        # 音节数 = 连字符数量 + 1
        return hyphenated.count('-') + 1
    except Exception as e:
        # 如果Pyphen处理失败，使用简单的回退方法
        print(f"Pyphen处理单词'{word}'时出错: {e}")
        return _fallback_syllable_count(word)

def _fallback_syllable_count(word):
    """回退的音节计数方法（主要适用于英语）"""
    word = word.lower()
    if len(word) <= 3:
        return 1
        
    count = 0
    vowels = "aeiouy"
    
    if word[0] in vowels:
        count += 1
        
    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels:
            count += 1
            
    if word.endswith('e'):
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
        
    return max(1, count)

def cal_syllable_details(text, lang='en'):
    """
    提供详细的音节分析信息
    
    返回:
    {
        'total_syllables': 总音节数,
        'word_count': 单词数,
        'syllables_per_word': 平均每个单词的音节数,
        'syllable_breakdown': 每个单词的音节分解
    }
    """
    if not text or not text.strip():
        return {'total_syllables': 0, 'word_count': 0, 'syllables_per_word': 0, 'syllable_breakdown': []}
    
    cleaned_text = re.sub(r'[^\w\s]', ' ', text.strip())
    words = [word for word in cleaned_text.split() if word and not word.isdigit()]
    
    if not words:
        return {'total_syllables': 0, 'word_count': 0, 'syllables_per_word': 0, 'syllable_breakdown': []}
    
    breakdown = []
    total_syllables = 0
    
    for word in words:
        syllable_count = cal_syllable_count(word, lang)
        breakdown.append({'word': word, 'syllables': syllable_count})
        total_syllables += syllable_count
    
    word_count = len(words)
    avg_syllables = round(total_syllables / word_count, 2) if word_count > 0 else 0
    
    return {
        'total_syllables': total_syllables,
        'word_count': word_count,
        'syllables_per_word': avg_syllables,
        'syllable_breakdown': breakdown
    }


# ========== 日语音节计数相关函数 ==========

# 数字到日文假名的映射
_DIGIT_TO_KANA = {
    '0': 'ゼロ', '1': 'いち', '2': 'に', '3': 'さん', '4': 'よん',
    '5': 'ご', '6': 'ろく', '7': 'なな', '8': 'はち', '9': 'きゅう'
}

# 初始化日语形态分析器
_tagger = fugashi.Tagger()


def _digit_token_to_kana(token):
    """
    将纯数字token逐位转换为日文假名。
    例如: "123" -> "いちにさん"
    """
    return ''.join(_DIGIT_TO_KANA.get(ch, ch) for ch in token)


def _count_japanese_mora(token):
    """
    计算日文字符串（假名）的音拍数。
    处理促音、拨音、长音、拗音等特殊规则。
    如果没有假名（例如纯中文），返回字符数。
    """
    has_kana = any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in token)
    if not has_kana:
        # 纯中文或其他，返回字符数
        return len(token)
    
    mora_count = 0
    i = 0
    length = len(token)

    while i < length:
        char = token[i]
        # 检查是否是拗音的一部分（例如きゃ、きゅ、きょ、しゃ等）
        if i + 1 < length and token[i+1] in 'ゃゅょャュョ':
            # 拗音整体算1拍
            mora_count += 1
            i += 2  # 跳过两个字符
        # 检查是否是促音、拨音、长音（它们各算1拍）
        elif char in 'っッんンー':
            mora_count += 1
            i += 1
        # 检查是否是其他假名（清音、浊音、半浊音等，每个通常算1拍）
        elif '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff':  # 平假名或片假名范围
            mora_count += 1
            i += 1
        else:
            # 非日文字符，跳过（或可根据需要处理）
            i += 1

    return mora_count


def _japanese_syllable_count(text):
    """
    使用fugashi进行精确的日语音拍计算。
    此函数解析文本，获取每个词的读音（假名），再计算音拍。
    适用于包含日语汉字的文本。
    """
    total_mora = 0
    # 使用fugashi解析句子
    parsed_nodes = _tagger(text)
    for word in parsed_nodes:
        # 尝试获取词的读音（假名形式），优先级: kana -> pronBase -> surface
        reading = getattr(word.feature, 'kana', None)
        if reading is None:
            reading = getattr(word.feature, 'pronBase', None)
        if reading is None:
            reading = word.surface  # 最后回退到表面形式

        # 计算该读音的音拍数
        word_mora = _count_japanese_mora(reading)
        total_mora += word_mora

    return total_mora





# 测试示例
if __name__ == "__main__":
    # 原有测试用例
    test_cases = [
        ("你好世界 Hello World", "zh"),  # 中英混合
        ("Hello World", "en"),
        ("The quick brown fox jumps over the lazy dog", "en"),
        ("Der schnelle braune Fuchs springt über den faulen Hund", "de"),
        ("Le renard brun rapide saute par-dessus le chien paresseux", "fr"),
    ]
    
    # 新增：日语测试用例（展示改进效果）
    japanese_test_cases = [
        "こんにちは",        # 5音拍 (ko-n-ni-chi-ha)
        "きょう",            # 2音拍 (kyo-u) - 拗音测试
        "きっと",            # 3音拍 (ki-t-to) - 促音测试
        "お母さん",          # 5音拍 (o-ka-a-sa-n) - 汉字+假名
        "コーヒー",          # 4音拍 (ko-o-hi-i) - 长音测试
        "りょこう",          # 4音拍 (ryo-ko-u) - 拗音测试
        "東京",              # 5音拍 (to-u-kyo-u) - 汉字
        "123",              # 8音拍 (i-chi-ni-sa-n)
        "Hello世界",         # 5音拍 (Hello 2 + 世界 3)
        "一緒に勉強しましょう",  # 日语汉字+假名混合
    ]
    
    print("=" * 70)
    print("音节计数测试结果 (多语言支持)")
    print("=" * 70)
    
    for text, lang in test_cases:
        count = cal_syllable_count(text, lang)
        details = cal_syllable_details(text, lang)
        
        print(f"\n语言: {lang}")
        print(f"文本: '{text}'")
        print(f"总音节数: {count}")
        print(f"单词数: {details['word_count']}")
        print(f"平均音节/词: {details['syllables_per_word']}")
        
        if details['syllable_breakdown'] and len(details['syllable_breakdown']) <= 10:
            print("详细分解:")
            for item in details['syllable_breakdown']:
                print(f"  '{item['word']}': {item['syllables']} 音节")
        print("-" * 70)
    
    # 日语专项测试
    print("\n" + "=" * 70)
    print("日语音拍计数专项测试（使用 fugashi 精确计算）")
    print("=" * 70)
    
    for text in japanese_test_cases:
        count = cal_syllable_count(text, 'ja')
        print(f"'{text}' -> {count} 音拍")
    
    print("\n" + "=" * 70)
        