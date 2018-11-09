#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
有道智云 (http://ai.youdao.com/) 翻译功能的简单封装
"""

from hashlib import md5
from random import randint

import fire
import requests
from dotenv import load_dotenv
from envs import env

__all__ = ['URL', 'ERRORS', 'translate', 'TranslateError']

URL = 'http://openapi.youdao.com/api'

ERRORS = dict((code, msg) for code, msg in (
    (101, '缺少必填的参数，出现这个情况还可能是et的值和实际加密方式不对应'),
    (102, '不支持的语言类型'),
    (103, '翻译文本过长'),
    (104, '不支持的API类型'),
    (105, '不支持的签名类型'),
    (106, '不支持的响应类型'),
    (107, '不支持的传输加密类型'),
    (108, 'appKey无效，注册账号， 登录后台创建应用和实例并完成绑定， 可获得应用ID和密钥等信息，其中应用ID就是appKey（ 注意不是应用密钥）'),
    (109, 'batchLog格式不正确'),
    (110, '无相关服务的有效实例'),
    (111, '开发者账号无效'),
    (113, 'q不能为空'),
    (201, '解密失败，可能为DES,BASE64,URLDecode的错误'),
    (202, '签名检验失败'),
    (203, '访问IP地址不在可访问IP列表'),
    (205, '请求的接口与选择的接入方式不一致'),
    (301, '辞典查询失败'),
    (302, '翻译查询失败'),
    (303, '服务端的其它异常'),
    (401, '账户已经欠费'),
    (411, '访问频率受限,请稍后访问'),
    (2005, 'ext参数不对'),
    (2006, '不支持的voice'),
))

LANG_AUTO = 'auto'
LANG_ZH = 'zh'


def translate(q, lang_from=LANG_AUTO, lang_to=LANG_ZH, appkey='', appsecret=''):
    """调用有道智云翻译API

    Parameters
    ----------
    q : str
        请求翻译 query. UTF-8编码
    lang_from : str
        翻译源语言(default=auto). 可设置为auto，见 `语言列表 <http://ai.youdao.com/docs/doc-trans-api.s>`_
    lang_to : str
        译文语言(default=zh). **不** 可设置为auto，见 `语言列表 <http://ai.youdao.com/docs/doc-trans-api.s>`_
    appkey : str
        APP ID (默认：环境变量 YOUDAO_FANYI_APP_KEY)
    appsecret : str
        APP Secret (默认：环境变量 YOUDAO_FANYI_APP_SECRET)

    Raises
    ------
    TranslateError
        服务器返回的翻译错误。相见有道翻译API错误说明文档

    Returns
    -------
    str
        翻译结果
    """
    load_dotenv()
    if not appkey:
        appkey = env('YOUDAO_FANYI_APP_KEY').strip()
    if not appsecret:
        appsecret = env('YOUDAO_FANYI_APP_SECRET').strip()

    q = q.strip()

    salt = '{0}'.format(randint(10000, 99999))
    s = '{0}{1}{2}{3}'.format(appkey, q, salt, appsecret)
    sign = md5(s.encode()).hexdigest().upper()

    r = requests.get(
        URL,
        params={
            'q': q,
            'from': lang_from.strip(),
            'to': lang_to.strip(),
            'appKey': appkey.strip(),
            'salt': salt,
            'sign': sign,
        }
    )
    r.raise_for_status()
    ret_obj = r.json()
    error_code = int(ret_obj.get('errorCode', 0))
    if error_code:
        raise TranslateError(error_code)
    return ret_obj['translation'][0]


class TranslateError(Exception):
    def __init__(self, code):  # type: (int)->TranslateError
        message = ERRORS.get(code, '')
        if not message:
            message = 'Error {0}'.format(code)
        super().__init__(message)
        self._code = code
        self._message = message

    @property
    def code(self):  # type: ()->int
        return self._code

    @property
    def message(self):  # type: ()->str
        return self._message


if __name__ == '__main__':
    fire.Fire(translate)
