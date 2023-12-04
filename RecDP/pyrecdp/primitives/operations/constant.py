import string
import emoji

# special characters
MAIN_SPECIAL_CHARACTERS = string.punctuation + string.digits \
                          + string.whitespace
OTHER_SPECIAL_CHARACTERS = (
    "    　    ￼’“”–ー一▬…✦�­£​•€«»°·═"
    "×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
    "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚"
    "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖"
    "」﴾》"
)
EMOJI = list(emoji.EMOJI_DATA.keys())
SPECIAL_CHARACTERS = set(MAIN_SPECIAL_CHARACTERS + OTHER_SPECIAL_CHARACTERS)
SPECIAL_CHARACTERS.update(EMOJI)

HF_TOKENIZER = 'EleutherAI/pythia-6.9b-deduped'

VARIOUS_WHITESPACES = {
    ' ', '	', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', '　', '​', '‌', '‍', '⁠', '￼', ''
}

DEFAULT_HEADER = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng, \
            */*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, \
            like Gecko) Chrome/113.0.0.0 Safari/537.36'
}
