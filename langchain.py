{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "30ZXnJREIGjd"
   },
   "source": [
    "# LangChain 기초 실습"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "mcfOGd0_IGjf"
   },
   "source": [
    "## 1. langchain 패키지 설치\n",
    "- openai api는 파이썬용 SDK 를 설치하거나 HTTP Client 를 사용하여 REST API 를 직접 호출하는 방식으로 사용할 수 있습니다.\n",
    "- 프로그래밍 언어 선택에 제약이 있는 경우를 제외하면 대부분 파이썬 SDK 를 설치하여 사용합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%capture\n",
    "!pip install langchain openai"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Azure OpenAI 리소스 정보 설정\n",
    "- openai API 를 호출하기 위해서는 API key 를 입력해야합니다.\n",
    "- Azure OpenAI 의 경우, API key 뿐만 아니라 endpoint, deployment name 등이 추가로 필요합니다.\n",
    "- 이 실습 환경에서는 Azure OpenAI 를 사용하기 위해 필요한 정보를 환경변수로 제공합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#실습용 AOAI 환경변수 읽기\n",
    "import os\n",
    "\n",
    "AOAI_ENDPOINT=os.getenv(\"AOAI_ENDPOINT\")\n",
    "AOAI_API_KEY=os.getenv(\"AOAI_API_KEY\")\n",
    "AOAI_DEPLOY_GPT4O=os.getenv(\"AOAI_DEPLOY_GPT4O\")\n",
    "AOAI_DEPLOY_GPT4O_MINI=os.getenv(\"AOAI_DEPLOY_GPT4O_MINI\")\n",
    "AOAI_DEPLOY_EMBED_3_LARGE=os.getenv(\"AOAI_DEPLOY_EMBED_3_LARGE\")\n",
    "AOAI_DEPLOY_EMBED_3_SMALL=os.getenv(\"AOAI_DEPLOY_EMBED_3_SMALL\")\n",
    "AOAI_DEPLOY_EMBED_ADA=os.getenv(\"AOAI_DEPLOY_EMBED_ADA\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. 간단한 Chat Completion 구현\n",
    "- AzureOpenAI 클라이언트를 생성하고 chat completions create 를 호출하여 간단히 GPT 의 응답을 받을 수 있습니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "AIMessage(content='저는 프로그래밍을 사랑합니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 31, 'total_tokens': 40, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_ded0d14823', 'prompt_filter_results': [{'prompt_index': 0, 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'filtered': False, 'detected': False}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}], 'finish_reason': 'stop', 'logprobs': None, 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'protected_material_code': {'filtered': False, 'detected': False}, 'protected_material_text': {'filtered': False, 'detected': False}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}, id='run-d3bc54f3-ab42-4c0f-8edb-040dece321cd-0', usage_metadata={'input_tokens': 31, 'output_tokens': 9, 'total_tokens': 40, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from langchain_core.messages import HumanMessage\n",
    "from langchain_openai import AzureChatOpenAI\n",
    "\n",
    "model = AzureChatOpenAI(\n",
    "    azure_endpoint=AOAI_ENDPOINT,\n",
    "    azure_deployment=AOAI_DEPLOY_GPT4O_MINI,\n",
    "    api_version=\"2024-10-21\",\n",
    "    api_key=AOAI_API_KEY\n",
    ")\n",
    "\n",
    "messages = [\n",
    "    (\"system\", \"You are a helpful assistant that translates English to Korean. Translate the user sentence.\",),\n",
    "    (\"human\", \"I love programming.\"),\n",
    "]\n",
    "ai_msg = model.invoke(messages)\n",
    "ai_msg"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- LangChain 은 LLM 의 답변을 AIMessage라는 객체 형태로 출력합니다.\n",
    "- content에는 실제 답변이 저장되며, response_metadata를 통해 토큰 사용량이나 필터링 정보 등을 추가로 얻을 수 있습니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "저는 프로그래밍을 사랑합니다.\n"
     ]
    }
   ],
   "source": [
    "print(ai_msg.content)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'},\n",
      "                            'protected_material_code': {'detected': False,\n",
      "                                                        'filtered': False},\n",
      "                            'protected_material_text': {'detected': False,\n",
      "                                                        'filtered': False},\n",
      "                            'self_harm': {'filtered': False,\n",
      "                                          'severity': 'safe'},\n",
      "                            'sexual': {'filtered': False, 'severity': 'safe'},\n",
      "                            'violence': {'filtered': False,\n",
      "                                         'severity': 'safe'}},\n",
      " 'finish_reason': 'stop',\n",
      " 'logprobs': None,\n",
      " 'model_name': 'gpt-4o-mini-2024-07-18',\n",
      " 'prompt_filter_results': [{'content_filter_results': {'hate': {'filtered': False,\n",
      "                                                                'severity': 'safe'},\n",
      "                                                       'jailbreak': {'detected': False,\n",
      "                                                                     'filtered': False},\n",
      "                                                       'self_harm': {'filtered': False,\n",
      "                                                                     'severity': 'safe'},\n",
      "                                                       'sexual': {'filtered': False,\n",
      "                                                                  'severity': 'safe'},\n",
      "                                                       'violence': {'filtered': False,\n",
      "                                                                    'severity': 'safe'}},\n",
      "                            'prompt_index': 0}],\n",
      " 'system_fingerprint': 'fp_ded0d14823',\n",
      " 'token_usage': {'completion_tokens': 9,\n",
      "                 'completion_tokens_details': {'accepted_prediction_tokens': 0,\n",
      "                                               'audio_tokens': 0,\n",
      "                                               'reasoning_tokens': 0,\n",
      "                                               'rejected_prediction_tokens': 0},\n",
      "                 'prompt_tokens': 31,\n",
      "                 'prompt_tokens_details': {'audio_tokens': 0,\n",
      "                                           'cached_tokens': 0},\n",
      "                 'total_tokens': 40}}\n"
     ]
    }
   ],
   "source": [
    "from pprint import pprint\n",
    "pprint(ai_msg.response_metadata)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Multi-Modal (GPT-Vision) 구현\n",
    "- o1, GPT-4o, GPT-4o-mini 및 GPT-4 Turbo with Vision 모델은 이미지를 분석하여 자연어로 답변할 수 있습니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<img src=\"https://img.animalplanet.co.kr/news/2025/01/09/700/tx7yr5v253qpymae44c4.jpg\" width=\"400\"/>"
      ],
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#이미지 확인\n",
    "from IPython.display import display, Image\n",
    "\n",
    "image_url = \"https://img.animalplanet.co.kr/news/2025/01/09/700/tx7yr5v253qpymae44c4.jpg\"\n",
    "display(Image(url=image_url, width=400))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "이 이미지는 귀엽고 사랑스러운 고양이가 편안하게 누워서 잠을 자고 있는 모습입니다. 고양이는 두 가지 색깔의 털을 가지고 있는데, 머리와 부분적인 몸이 주황색이고 나머지 부분은 흰색입니다. 특히 고양이는 몹시 편안해 보이며, 눈을 감고 미소 짓는 듯한 모습이 매력적입니다. 목에는 노란색 꽃 장식의 목걸이를 하고 있어 더욱 귀여운 느낌을 줍니다. 배경은 회색의 패브릭 소파로, 고양이의 털색과 잘 어울리며 아늑한 분위기를 연출하고 있습니다. 전체적으로 이 이미지는 평화롭고 따뜻한 느낌을 줍니다.\n"
     ]
    }
   ],
   "source": [
    "message = HumanMessage(\n",
    "    content=[\n",
    "        {\"type\": \"text\", \"text\": \"이미지를 설명해줘\"},\n",
    "        {\"type\": \"image_url\", \"image_url\": {\"url\": image_url}},\n",
    "    ],\n",
    ")\n",
    "response = model.invoke([message])\n",
    "print(response.content)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Message 객체의 종류 \n",
    "\n",
    "- **SystemMessage**: AI 모델의 동작을 조정하는 역할로, 대화의 컨텍스트나 톤을 설정하는 메시지. 일부 모델에서는 시스템 메시지를 별도 API 파라미터로 전달.\n",
    "\n",
    "- **HumanMessage**: 사용자가 입력한 내용을 나타내는 메시지로, 주로 텍스트 기반이지만 일부 모델은 이미지, 오디오 등의 멀티모달 데이터도 지원.\n",
    "\n",
    "- **AIMessage**: 모델의 응답을 나타내며, 텍스트뿐만 아니라 도구 호출 요청 등의 데이터를 포함할 수 있음. 응답 메타데이터와 토큰 사용량 정보 등을 포함하기도 함.\n",
    "\n",
    "- **AIMessageChunk**: AIMessage의 스트리밍 버전으로, 실시간으로 응답을 받을 수 있도록 청크 단위로 전송되며, 여러 청크를 합쳐 최종 메시지를 생성 가능.\n",
    "\n",
    "- **ToolMessage**: 도구 호출 결과를 포함하는 메시지로, 호출된 도구의 ID 및 실행 결과와 관련된 추가 데이터(artifact 등)를 저장.\n",
    "\n",
    "각 메시지는 역할(role), 콘텐츠(content), 메타데이터(metadata) 등의 정보를 포함하며, LangChain은 이를 다양한 챗 모델에서 공통적으로 사용할 수 있도록 표준화하였습니다.\n",
    "\n",
    "(https://python.langchain.com/docs/concepts/messages/)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. LCEL\n",
    "\n",
    "LCEL(LangChain Expression Language)은 LangChain에서 체인(Chain)과 프롬프트(Prompt)를 보다 직관적이고 유연하게 조합할 수 있도록 지원하는 언어입니다.\n",
    "\n",
    "LLM 에 프롬프트를 입력하고 출력 데이터를 정제하는 일련의 과정을 하나의 파이프라인으로 묶을 수 있습니다.\n",
    "\n",
    "1. **PromptTemplate**  \n",
    "   - 입력을 동적으로 받아 특정 형식의 프롬프트를 생성합니다.\n",
    "\n",
    "2. **LLM**  \n",
    "   - LLM 객체가 될 수도 있고 또는 chain 끼리 연결할 수도 있습니다.\n",
    "\n",
    "3. **Output Parser**\n",
    "   - AIMessage 를 원하는 포맷으로 변환하여 받을 수 있습니다.\n",
    "   - StrOutputParser\n",
    "   - JsonOutputParser\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PromptTemplate(input_variables=['text'], input_types={}, partial_variables={}, template='{text} 를 영어로 번역해주세요.')"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from langchain.prompts import PromptTemplate\n",
    "\n",
    "prompt = PromptTemplate.from_template(\"{text} 를 영어로 번역해주세요.\")\n",
    "prompt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'한국의 수도는 서울입니다. 를 영어로 번역해주세요.'"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prompt.format(text=\"한국의 수도는 서울입니다.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "AIMessage(content='\"The capital of South Korea is Seoul.\"', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 8, 'prompt_tokens': 20, 'total_tokens': 28, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_b705f0c291', 'prompt_filter_results': [{'prompt_index': 0, 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'filtered': False, 'detected': False}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}], 'finish_reason': 'stop', 'logprobs': None, 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'protected_material_code': {'filtered': False, 'detected': False}, 'protected_material_text': {'filtered': False, 'detected': False}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}, id='run-2e0a31eb-c65b-4ade-a665-21095f2c1836-0', usage_metadata={'input_tokens': 20, 'output_tokens': 8, 'total_tokens': 28, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "chain1 = prompt | model\n",
    "chain1.invoke({\"text\": \"한국의 수도는 서울입니다\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_core.output_parsers import StrOutputParser\n",
    "\n",
    "output_parser = StrOutputParser()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\"The capital of South Korea is Seoul.\"'"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "chain2 = prompt | model | output_parser\n",
    "chain2.invoke({\"text\": \"한국의 수도는 서울입니다\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'origin': '한국의 수도는 서울입니다',\n",
       " 'translated': 'The capital of South Korea is Seoul.'}"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from langchain_core.output_parsers import JsonOutputParser\n",
    "from pydantic import BaseModel, Field\n",
    "\n",
    "class Translate(BaseModel):\n",
    "    origin: str = Field(description=\"번역 전 텍스트\")\n",
    "    translated: str = Field(description=\"번역된 텍스트\")\n",
    "\n",
    "json_parser = JsonOutputParser(pydantic_object=Translate)\n",
    "\n",
    "json_prompt = PromptTemplate(\n",
    "    template=\"사용자의 text를 영어로 번역하세요.\\n{format_instructions}\\n{text}\\n\",\n",
    "    input_variable=[\"text\"],\n",
    "    partial_variables={\"format_instructions\": json_parser.get_format_instructions()}\n",
    ")\n",
    "\n",
    "chain3 = json_prompt | model | json_parser\n",
    "chain3.invoke({\"text\": \"한국의 수도는 서울입니다\"})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## [부록] Structured Outputs\n",
    "\n",
    "JsonOutputParser 는 프롬프트를 기반으로 LLM 이 Json 으로 답변한 것을 파싱하여 Json 값으로 반환해주지만 LLM 이 실수할 수 있습니다.\n",
    "\n",
    "Structured Output 은 LLM 이 JSON 스키마를 준수하여 답변하도록 OpenAI API가 보장해줍니다.\n",
    "\n",
    "- 신뢰할 수 있는 유형 안전성: 잘못 포맷된 응답을 검증하거나 다시 시도할 필요가 없습니다.\n",
    "- 명시적 거부: 안전 기반 모델 거부는 이제 프로그래밍 방식으로 감지할 수 있습니다.\n",
    "- 더 간단한 프롬프트: 일관된 형식을 달성하기 위해 강력한 표현의 프롬프트가 필요하지 않습니다.\n",
    "\n",
    "https://python.langchain.com/docs/concepts/structured_outputs/\n",
    "\n",
    "https://platform.openai.com/docs/guides/structured-outputs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Translate(origin='한국의 수도는 서울입니다.', translated='The capital of Korea is Seoul.')"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "class Translate(BaseModel):\n",
    "    origin: str = Field(description=\"번역 전 텍스트\")\n",
    "    translated: str = Field(description=\"번역된 텍스트\")\n",
    "\n",
    "model_with_structure = model.with_structured_output(Translate)\n",
    "model_with_structure.invoke(\"한국의 수도는 서울입니다. 를 영어로 번역해줘\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. ChatPromptTemplate\n",
    "- 대화 히스토리를 프롬프트로 입력할 때 활용합니다.\n",
    "- \"system\" : 전역에 반영되는 시스템 프롬프트\n",
    "- \"human\" : 사용자 입력 메시지\n",
    "- \"ai\" : LLM 의 답변 메시지"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[SystemMessage(content='당신은 번역 업무를 수행하는 에이전트입니다. 당신의 이름은 선경 입니다.', additional_kwargs={}, response_metadata={}),\n",
       " HumanMessage(content='반가워!', additional_kwargs={}, response_metadata={}),\n",
       " AIMessage(content='안녕하세요, 반갑습니다.', additional_kwargs={}, response_metadata={}),\n",
       " HumanMessage(content='너의 이름은 뭐야?', additional_kwargs={}, response_metadata={})]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from langchain_core.prompts import ChatPromptTemplate\n",
    "\n",
    "chat_prompt = ChatPromptTemplate.from_messages(\n",
    "    [\n",
    "        (\"system\", \"당신은 번역 업무를 수행하는 에이전트입니다. 당신의 이름은 {name} 입니다.\"),\n",
    "        (\"human\", \"반가워!\"),\n",
    "        (\"ai\", \"안녕하세요, 반갑습니다.\"),\n",
    "        (\"human\", \"{user_input}\")\n",
    "    ]\n",
    ")\n",
    "\n",
    "chat_prompt.format_messages(\n",
    "    user_input=\"너의 이름은 뭐야?\",\n",
    "    name=\"선경\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'제 이름은 선경입니다. 당신과 대화하게 되어 기쁩니다!'"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "chain4 = chat_prompt | model | StrOutputParser()\n",
    "chain4.invoke({\"user_input\": \"너의 이름은 뭐야?\", \"name\": \"선경\"})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. MessagePlaceholder\n",
    "- 채팅 기록이나 특정 메시지 데이터를 동적으로 프롬프트에 넣을 때 사용합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'물론이죠! 다양한 메뉴로 추천해드릴게요:\\n\\n1. **비빔밥** - 신선한 채소와 고기를 밥과 함께 비벼 먹는 맛있는 한 그릇 요리입니다.\\n2. **김치찌개** - 따끈하고 얼큰한 김치찌개에 밥과 함께 즐기면 좋습니다.\\n3. **볶음밥** - 냉장고에 남은 재료를 활용해 볶음밥을 만들어보세요! 간단하지만 맛있습니다.\\n4. **족발** - 쫄깃한 족발을 추천해드려요! 쌈채소와 함께 먹으면 더욱 맛있습니다.\\n5. **샐러드와 파스타** - 가벼운 식사를 원하시면 신선한 샐러드와 함께 크림 또는 토마토 소스 파스타를 조합해보세요.\\n\\n오늘은 어떤 메뉴가 마음에 드세요?'"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from langchain_core.prompts import MessagesPlaceholder\n",
    "\n",
    "chat_prompt2 = ChatPromptTemplate.from_messages([\n",
    "    (\"system\", \"You are a helpful assistant.\"),\n",
    "    MessagesPlaceholder(variable_name=\"chat_history\"),  # 이전 대화 기록을 저장하는 플레이스홀더\n",
    "    (\"human\", \"{input}\")  # 사용자 입력\n",
    "])\n",
    "\n",
    "chain5 = chat_prompt2 | model | StrOutputParser()\n",
    "\n",
    "messages = [\n",
    "    {\"role\": \"human\", \"content\": \"오늘 점심메뉴 추천해줘!\"},\n",
    "    {\"role\": \"ai\", \"content\": \"해장국 어떠신가요?\"}\n",
    "]\n",
    "\n",
    "chain5.invoke({\"chat_history\": messages, \"input\": \"다른 메뉴는 없을까?\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
