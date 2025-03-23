<p align="center">
  <img src="https://www.merchandisingplaza.co.uk/282130/2/Stickers-Star-Trek-STAR-TREK-Spock-Live-Long-Prosper-Sticker-l.jpg" 
       alt="Live Long and Prosper Sticker" 
       width="300">
</p>


# HW2 generative model based чат бот

**Задание**: необходимо разработать чат бот, используя генеративный подход. Бот должен вести диалог как определенный персонаж сериала, имитируя стиль и манеру конкретного персонажа сериала. Важно учесть особенности речи и темы, которые поднимает персонаж, его типичные реакции.

## Сбор данных
В качестве основы для чат бота я взял скрипты к сериалам "Star Trek", которые загрузил из репозитория по [ссылке](https://github.com/varenc/star_trek_transcript_search), в частности реплики Мистера Спока, члена экипажа корабля, ученого с планеты Вулкан.


### Данные для retrieval-элемента чат-бота

Первоначальная обработка данных схожа с той, что я делал для предыдущего домашнего задания при подготовке retrieval-based чат бота. 

Данные обрабатываются следующим образом:
- очистка скрипта
- отбор реплик персонажа в качестве ответов. Именно этими репликами будет пользоваться бот для формирования своего ответа на высказывание пользователя.
- выделение предшествующей фразы как вопроса. Если это фраза первая в сцене, то это поле будет пустым.
- отбор предыдущих реплик как контекста диалога. Если фраза первая в сцене, то контекст также будет пустым. Контекст - идущие подряд предложения.

Для улучшения фактологической связанности ответов были использованы контекстно-зависимык эмбеддинги. Для сохранения тематики и стиля буду дополненять генерацию retrieval данными в зависимости от косинусной близости к вопросу с контекстом от пользователя, и, как и в предыдущем задании, все начальные данные векторизую в базу данных (файл ```spock_lines_vectorized.pkl```). Для векторизации я буду использовать обученную в предыдущем задании модель bi-encoder, которую сохранил в мой репозиторий на Hugging Face ([ссылка](https://huggingface.co/greatakela/gnlp_hw1_encoder)). Детально про обучение этой модели - в предыдущем задании ([ссылка на репозиторий](https://github.com/greatakela/ChatBot/tree/main))

### Данные для генеративной модели

Для обучение генеративной модели я буду использовать первоначальную переработку как и для retrieval-данных, описанную выше. Дополнительно, я разбью весь полученный контекст на части для его аугментации: например, если в контексте есть три предложения, то я в итоге получаю 4 семпла для данных для одной реплики:

- ответ + вопрос + предложение 3 +  предложение 2 + предложение 1
- ответ + вопрос + предложение 3 +  предложение 2
- ответ + вопрос + предложение 3
- ответ + вопрос

Таким образом, получается около 38 тыс. семплов для обучения. Обработанные и дополненные таким образом данные сохранены в файл ```spock_lines_context.pkl```.

Код подготовки данных в ноутбуке [здесь](https://github.com/greatakela/GenChatBot/blob/main/Notebooks/GNLP_HW2_data_prep.ipynb)

## Архитектура чат бота

Схематично процесс работы чат бота представлен на рисунке ниже.

![image](https://github.com/greatakela/GenChatBot/blob/main/static/ArchGenBot.png)

### Retrieval-часть чат бота

**База данных реплик** - векторизованные при помощи модели [обученного энкодера](https://huggingface.co/greatakela/gnlp_hw1_encoder) скрипты, включающие контекст и вопрос. Детальное описание процесса обучения я приводил в предыдущем задании. Здесь же просто воспользуюсь готовой моделью из моего репозитория на Hugging Face ([ссылка](https://huggingface.co/greatakela/gnlp_hw1_encoder)).

Реплика из базы данных, вопрос и контекст, которые максимально похожи на текущий запрос от пользователя (только top-1 для ускорения работы), будут подаваться на вход генеративной языковой модели как часть контекста при реализации стратегии **RAG (retrieval-augmented generation)** для того, чтобы немного добавить фактов и деталей из оригинального скрипта сериала в диалоге с пользователем.

Для подачи retrieval-реплик в генеративную модель я установил порог в **0.9** по косинусной близости. Если топовая реплика имеют меньшую косинусную близость с контекстом-вопросом из диалога пользователя, то она не подается в генеративную модель. Если больше, то подается.

### Generative-часть чат бота
Основной частью чат-бота является генеративная модель. Имея значительное количество исходных данных, я попробовал дообучить небольшую модель семейства T5, которая известна своей универсальностью. В качестве основной для своего эксперимента я выбрал модель ```google/flan-t5-base``` на 248 млн. параметров (детально про модель можно посмотреть [здесь](https://huggingface.co/google/flan-t5-base) в репозитории Hugging Face). Эта модель хорошо улавливает семантическую связь между словами в тексте и может использоваться для предобработки входных данных или как часть ансамбля моделей. Я обучал модель 5 эпох на платформе Colab с иcпользованием A100. Ноутбук с обучением можно посмотреть [здесь](https://github.com/greatakela/GenChatBot/blob/main/Notebooks/GNLP_HW2_FLAN_T5_train_model.ipynb).

В модель подаются данные контекста и вопроса в качестве фичей, ответ рассматривается в качестве target. При этом вопрос и контекст соединяются в промпт типа ``` "context: " + контекст + "</s>" + 'question: ' + вопрос ```.

#### Оценка обучения

Во время обучения логировались такие стандартные метрики как train/eval loss. Дополнительно я добавил автоматические метрики для сравнения похожести генерируемых текстов ответа с ответом из скрипта. Использование автоматических метрик для оценки генерации довольно часто критикуется и не может рассматриваться в качестве достаточных. В моем случае я использовал их как некое направление, чтобы понять, нужно ли учить модель дальше. Данные метрики конечно же нужно дополнять дополнительными human-based оценками уже на этапе подбора стратегии генерации. В качестве автоматических метрик я воспользовался библиотекой ```evaluate``` на Hugging Face и выбрал из нее метрики [**rouge**](https://huggingface.co/spaces/evaluate-metric/rouge) и [**bertscore**](https://huggingface.co/spaces/evaluate-metric/bertscore), которые представляют из себя пакеты метрик, включающие:

- **rouge 1** - соответствие между сгенерированным текстом и таргетом на основе unigram (чем выше, тем больше похожи тексты)
- **rouge 2** - соответствие между сгенерированным текстом и таргетом на основе 2-gram (чем выше, тем больше похожи тексты)
- **rouge L** - измеряет соответствие самой долгой последовательности между сгенерированным текстом и таргетом (чем выше, тем больше похожи тексты)
- **rouge average generated length** - средняя длина генерируемого текста (нужно для понимания как работает модель)
- **bertscore recall** - косинусная близость между векторами таргета и генерируемого текста, оцениваемая с помощью эмбедингов модели bert (чем ближе к 1, тем более похожие тексты)
- **bertscore precision** - косинусная близость между векторами генерируемого текста и таргета, оцениваемая с помощью эмбедингов модели bert (чем ближе к 1, тем более похожие тексты)
- **bertscore f1** - f1 для предыдущих метрик (чем ближе к 1, тем более похожие тексты)

Ниже принт-скрины из wandb с графиками изменения метрик в процессе обучения:

<img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_bs_r.png" width="32.5%"> <img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_bs_p.png" width="32.5%"> <img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_bs_f1.png" width="32.5%">

<img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_rouge_1.png" width="32%"> <img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_rouge_2.png" width="32%"> <img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_rouge_l.png" width="32%">

Продолжение:

<img src="https://github.com/greatakela/GenChatBot/blob/main/static/train_loss.png" width="49.5%"> <img src="https://github.com/greatakela/GenChatBot/blob/main/static/eval_loss.png" width="49.5%">

## Выводы по результатам модели:
Данные результаты указывают на очень высокую эффективность модели в процессе обучения и валидации. 
Значения потерь как на тренировочном, так и на валидационном наборах данных уменьшаются с каждой эпохой, что указывает на то, что модель успешно учится и адаптируется к задаче. Снижение потерь говорит о том, что модель становится все лучше в предсказании правильных ответов. Значительное уменьшение потерь на тренировочной выборке свидетельствует о хорошем обучении модели. Это указывает на то, что модель успешно извлекает закономерности из обучающих данных и эффективно обобщает эти знания. Потери на валидационной выборке также снижаются, но более плавно, чем на тренировочной. Это естественно, поскольку валидационный набор данных используется для оценки обобщающей способности модели на данных, которые она ранее не видела. Снижение валидационных потерь подтверждает, что модель не переобучается и хорошо обобщает знания на новых данных.
Наблюдаемая разница между тренировочными и валидационными потерями уменьшается с течением времени, что является положительным индикатором. Однако важно следить за этим различием: если оно становится слишком маленьким, это может указывать на недообучение, а если слишком большим — на переобучение.
Исходя из всего выше сказанного, можно подытожить, что уменьшение валидационных потерь и сближение их значений с тренировочными свидетельствует о хорошей обобщающей способности модели. Это говорит о том, что модель способна адекватно реагировать на новые данные, что является ключевым для задач генерации текста.
Из приведенных графиков видно, что у модели есть еще потенциал для fine-tune, так как продолжают уменьшаться eval и train loss.
Несмотря на дальнейший потенциал обучения я остановил обучения на 5 эпохах, так как метрики похожести текстов перестали скачкообразно меняться и вышли на стабильные, хотя и немного растущие значения.

#### Подбор стратегии генерации

Для определения параметров генерации для чат-бота проведу несколько экспериментов с моделью, меняя параметры генерации. Ноутбук с экспериментами можно посмотреть вот [здесь](https://github.com/greatakela/GenChatBot/blob/main/Notebooks/GNLP_HW2_generation_evaluation.ipynb). В качестве неизменных параметров (после проверки) я выбрал:

- do_sample=True - вносим больший элемент рандомности
- max_length=1000 - не ограничиваем генерации длиной
- repetition_penalty=2.0 - модель немного недоучена, поэтому приходится добавить штраф за повторения
- top_k=50 - если оставить параметр меньше, то модель плохо следит за репликами пользователя
- no_repeat_ngram_size=2 - продолжение борьбы с недоученностью модели
  
Экспериментировать я буду с параметрами ```top-p``` и ```temperature``` - оценю, как они влияют на повторяемость и креативность диалога. Оценивать буду генерации по косинусной близости между сгенерированным текстом и ответами из скрипта. Данные возьму из файла ```spock_lines_context.pkl```, сделаю из этого файла случайную выборку в размере 30 семплов и сравню ответ модели на семпл с таргетным ответом. В качестве bi-encoder возьму свою же модель, которую использую для ранжирования в retrieval-части чат-бота. Дополнительно посмотрю на время генерации. 

В качестве экспериментальных значений возьму следующие параметры:  
- **temperature = 0.2 top_p = 0.1** - ожидаю стандартные тексты, возможно без характеристик героя
- **temperature = 0.5 top_p = 0.5** - ожидаю стандартные тексты, немного больше свободы для генерации у модели
- **temperature = 0.7 top_p = 0.8** - больше креативности, начинают проявляться характерные черты героя 
- **temperature = 0.9 top_p = 0.9** - еще больше креативности, проявляются характерные черты героя
- **temperature = 1 top_p = 0.95** - возможен уход от контекста


<img src="https://github.com/greatakela/GenChatBot/blob/main/static/gen_time.png" width="49.5%"> <img src="https://github.com/greatakela/GenChatBot/blob/main/static/cos_sim.png" width="49.5%">

Если смотреть на косинусную близость (см. графики выше), то видно, что генерации при сочетании temp=1 top_p=0.95 чаще всего похожи на таргетные (показатель косинусной близости реже бывает ниже 0,6), т.е. лучше передают стиль персонажа, но при этом генерации чаще всего занимают больше времени.

Из сгенерированных текстов очень заметно, что повышение обоих параметров ведет к генерации более интересных и разнообразных текстов. Тексты с низкими параметрами выглядят довольно скучно и ожидаемо не передают характера персонажа. При высоких параметрах остается риск ухода от модели от контекста и придумывания собственных фактов. Интересно, что показатели косинусной близости не сильно отличаются, что подтверждает сделанные ранее выводы о том, что расчетные метрики при генерации текста нельзя использовать без оценки генераций человеком.

Финальные параметры для генерации - **temperature=0.9 и top_p=0.9**  - это позволит сохранить разнообразие генераций и уменьшить уровень "галлюцинаций".

## Структура репозитория

```bash
│   README.md - отчет по ДЗ 2
│   requirements.txt
│   .gitignore
│   __init__.py
│   generative_bot.py - основной файл алгоритма
│   utilities.py - вспомогательные функции
│   app.py - для запуска UI c flask
│
├───Notebooks - ноутбуки с обучением и оценкой модели
├───templates - оформление веб-интерфейса
│       chat.html
├───static - оформление веб-интерфейса
│       style.css
├───data
│       spock_lines_context.pkl - дополненные данные для обучения модели
│       spock_lines_vectorized.pkl - векторная база данных контекст-вопрос
│       spock_lines.pkl - исходные данные
```

## Веб-сервис
Чат реализован на основе Flask, запускается скриптом ```app.py```, который выстраивает графический интерфейс, создает инстант класса ChatBot, загружает файлы и модели. 

Для локальной установки проекта нужно склонировать репозиторий ```https://github.com/greatakela/GenChatBot.git```, создать среду, затем сделать установку ```pip install -r requirements.txt```. Чат бот запускается командой ```python app.py```, и открывается в локальном окне браузера на ```http://127.0.0.1:5000```.

### Асинхронность на уровне кода Flask-приложения
Асинхронность в платформе Flask обеспечивается добавлением asynchronous route handlers, которые позволяют использовать асинхронный режим на уровне обработки событий самого приложения с помощью ```async ``` и ```await ```. Когда запрос поступает в асинхронное представление, Flask запускает цикл обработки каждого из событий в отдельном потоке.

В моей реализации у Flask-приложения всего 2 события:
- построение интерфейса
- получение запроса и генерация ответа от пользователя (здесь не может быть асинхронности, так как нужно сперва получить вопрос, чтобы сгенерировать результат)

Для демонстрации асинхронности на уровне кода приложения я добавил подпрограмму, которую должна дождаться задача генерации и которая будет выполняться параллельно с ней - небольшой sleep:

```python
async def sleep():
    await asyncio.sleep(0.1)
    return 0.1

@app.route("/get", methods=["GET", "POST"])
async def chat():
    msg = request.form["msg"]
    input = msg
    await asyncio.gather(sleep(), sleep())
    return get_Chat_response(input)

```
Каждый запрос по-прежнему связывает одну задачу, даже для асинхронных представлений. Положительным моментом является то, что асинхронный код можно запускать в самом представлении, например, для выполнения нескольких одновременных запросов к базе данных и/или HTTP-запросов к внешнему API и т. д. **НО количество запросов, которые веб-приложение может обрабатывать одновременно, останется прежним**. Поэтому переходим к следующему пункту :)

### Многопроцессорность и асинхронность gunicorn
**Gunicorn**  - WSGI (Web-Server Gateway Interface) для UNIX используется для создания многопроцессорности (нескольких workers) и возможности работать с приложением нескольким пользователем одновременно. Использование ```gevent``` позволяет workers работать в асинхронном режиме и принимать несколько соединений на одного worker. При указании кол-ва соединений на 1 worker можно использовать указанное кол-во клонов.

Для запуска такого режима gunicorn нужно прописать в dockerfile: ```CMD ["gunicorn", "--timeout", "1000", "--workers", "2", "--worker-class", "gevent", "--worker-connections" , "100", "app:app", "-b", "0.0.0.0:5000"]```

При запуске image на ВМ будет загружено gunicorn с двумя рабочими процессами, плюс 50 асинхронных gevent процессов на синхронный процесс gunicorn (50 * 2 = 100). Пример логов ниже:

```bash
admin@bot:~$ sudo docker run -it --name chat -p 5000:5000 --rm shakhovak/hw2_bot
[2024-03-08 17:25:27 +0000] [1] [INFO] Starting gunicorn 21.2.0
[2024-03-08 17:25:27 +0000] [1] [INFO] Listening at: http://0.0.0.0:5000 (1)
# Теперь используется `gevent` 
[2024-03-08 17:25:27 +0000] [1] [INFO] Using worker: gevent
[2024-03-08 17:25:27 +0000] [7] [INFO] Booting worker with pid: 7
[2024-03-08 17:25:27 +0000] [8] [INFO] Booting worker with pid: 8
```
В идеале нужно запускать такое приложение еще и с использованием веб-сервера nginx для более устойчивой работы, но так как это учебный пример, я решил обойтись без него.

## Заключение
На основе проведенного анализа можно сделать вывод о высокой эффективности разработанной модели для задачи автоматизированного чат бота. Однако для обеспечения более глубокого понимания ее способностей и ограничений необходимо провести дополнительные эксперименты, включая тестирование на более разнообразном и объемном наборе данных, а также оценку способности модели к обобщению на новых примерах.

## Web сервис
Для запуска веб-сервиса я собрал проект в docker контейнер на локальном компьютере, создал виртуальный сервер на платформе Kamatera и развернул залитый на него docker контейнер. Чат доступен по адресу http://185.53.209.56:5000/ .
Удалось оптимизировать docker контейнер, его размер получился меньше 2 Гб. Параметры виртуального сервера - 2 x CPU, RAM - 2 Gb, disk space - 80 Gb (осталось с предыдущего ДЗ -более чем достаточно, буду уменьшать)
