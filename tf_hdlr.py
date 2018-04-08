"""
Перечень необходимых библиотек:
    
math: библиотека для проведения вычислений
matplotlib: библиотека для визуализации данных
numpy: библиотека для работы с многомерными массивами, ускоряющая реализацию вычислительных алгоритмов
pandas: библиотека для обработки и анализа данных, используется над модулем `numpy`
sklearn: библиотека для машинного обучения
tensorflow: библиотека для машинного обучения

Использования аппаратного ускорения GPU или TPU не требуется.
"""

"""Шаг 1. Предварительный анализ и очистка данных."""

#Импортирует необходимые библиотеки.

import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

"""
Загрузим интересующий нас набор данных [1], в данном случае представляющий 
собой csv-файл [2].

[1]: https://developers.google.com/machine-learning/glossary/#dataset
[2]: https://ru.wikipedia.org/wiki/CSV

Предполагается, что интересующий нас csv-файл расположен
в рабочей директории, адрес который можно получить при помощи:
    
import os
os.getcwd()

Используем команду pandas.read_csv [3],  чтобы создать 
pandas-структуру DataFrame [4], содержащую данные из загруженного файла.

[3]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
[4]: https://developers.google.com/machine-learning/glossary/#dataframe 

DataFrame является основной абстракцией библиотеки pandas и представляет собой
проиндексированный многомерный массив значений.

В файле присутствуют кириллические символы, поэтому также 
используем аргумент windows-1251 [5] для эксплицитного определения 
параметра `encoding`.

[5]: https://en.wikipedia.org/wiki/Windows-1251
"""

magnet_data = pd.read_csv("data.csv", encoding="windows-1251", sep=",")

"""
Ознакомимся с основными статистическими характеристиками набора данных 
при помощи команды pandas.DataFrame.describe() [6].

[6]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html
"""

magnet_data.describe()

"""
Предполагая, что используется набор данных, предоставленный мне для выполнения
тестового задания, мы видим следующее.

Максимальные значения нескольких столбцов значительно превышают 
среднеквадратическое отклонение, указывая на присутствие выбросов [7] в данных. 
Избавимся от них, исключив из набора все ряды, для которых 
абсолютное значение z-оценки [8] превышает 3 в любом из столбцев. 

[7]: https://developers.google.com/machine-learning/glossary/#outliers
[8]: https://ru.wikipedia.org/wiki/Z-%D0%BE%D1%86%D0%B5%D0%BD%D0%BA%D0%B0

Реализуем это при помощи анонимной функции.

Примечание: для расчета z-оценки также можно воспользоваться командой 
scipy.stats.zscore [9].

[9]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html
"""

# Очищает данные от выбросов. Исключает из набора все ряды, для которых 
# абсолютное значение z-оценки превышает 3 в любом из столбцев. 

magnet_data = magnet_data[magnet_data[
    ["n1", "n2", "n3", "n4", "n5", "n6", "n7",
    "n8", "n9", "n10", "n11", "n12", "n13", 
    "n14", "n15", "n16", "n17", "n18", "n19",
    "n20", "n21", "n22", "n23", "n24", "n25"]
    ].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]           # Рассчитывает абсолютное значение z-оценки и сравнивает его с 3.

magnet_data.describe()                                                           # Выводит статистические характеристики для очищенного набора данных.

"""Шаг 2. Обработка данных для использования в моделировании."""

"""
Определенные типы сортировки данных снижают эффективность метода 
стохастического градиентного спуска [10]. Предотвратим это, создав 
случайную пермутацию [11] индексов данных и реиндексировав [12] DataFrame с ее помощью.

[10]: https://developers.google.com/machine-learning/glossary/#stochastic_gradient_descent
[11]: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.permutation.html
[12]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reindex.html
"""

magnet_data = magnet_data.reindex(np.random.permutation(magnet_data.index))      # Реиндексирует набор данных, используя случайную перестановку его индексов.

"""
Чтобы передавать данные регрессору [13] TensorFlow, мы должны создать 
входную функцию [14].

Мы представляем набор данных в виде словаря numpy-массивов [15] и используем 
TensorFlow Dataset API [16], чтобы разбить его на корзины [17]. 
После этого, мы указываем число эпох [18] и создаем итератор [19], 
передающий следующую корзину данных в регрессор.

[13]: https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/examples/get_started/regression/linear_regression.py
[14]: https://developers.google.com/machine-learning/glossary/#input_function
[15]: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.array.html
[16]: https://www.tensorflow.org/programmers_guide/datasets
[17]: https://developers.google.com/machine-learning/glossary/#batch
[18]: https://developers.google.com/machine-learning/glossary/#epoch
[19]: https://www.tensorflow.org/versions/master/api_docs/python/tf/data/Iterator
"""

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
  
    """Тренирует регрессор на нескольких входящих переменных.
  
    Аргументы:
      features: входящие переменные, тип: 'DataFrame'
      targets: выходящие переменные, тип: 'DataFrame'
      batch_size: размер корзин, тип: 'int'
      shuffle: перемешивание данных, тип: 'bool'
      num_epochs: число эпох, тип: 'int', None = без ограничений
      
    Возвращает:
      Кортеж (входящие переменные, выходящие переменные), на основании которого 
      обновляются параметры, используемые со следующей корзиной данных.
    """
    
    # Представляет данные в виде словаря numpy-массивов.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Создает набор данных с заданным размером корзин и числом эпох.
    ds = Dataset.from_tensor_slices((features,targets)) # Ограничение: 2GB.
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Если параметр shuffle=True, перемешивает данные.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Передает регрессору следующую корзину данных.
    features, labels = ds.make_one_shot_iterator().get_next()
    
    return features, labels

"""
В наборе данных имеются нумерические [20] и категорические [21] столбцы. 
Последние, являясь данными строкового типа, не могут быть переданы алгоритму 
в качестве входящих переменных [22] напрямую. Мы не будем использовать их для 
обучения модели, но это возможно. Способ сделать это описан в Приложении 1, 
расположенном в конце описания данного шага.

[20]: https://developers.google.com/machine-learning/glossary/#numerical_data
[21]: https://developers.google.com/machine-learning/glossary/#categorical_data
[22]: https://developers.google.com/machine-learning/glossary/#feature

Создадим функцию, производящую предварительную обработку входящих переменных [23].
Для иллюстрации концепции синтетических переменных [24], сформируем одну из них 
в теле функции.

[23]: https://developers.google.com/machine-learning/glossary/#feature
[24]: https://developers.google.com/machine-learning/glossary/#synthetic_feature
"""

def preprocess_features(magnet_data):
  
  """Подготавливает входящие переменные.

  Аргументы:
    magnet_data: исходный набор данных, тип: 'DataFrame'
    
  Возвращает:
    Набор данных в виде pandas DataFrame, содержащий входящие переменные, 
    которые будут использоваться для обучения модели.
  """
  
  selected_features = magnet_data[
    ["n1", "n2", "n3", "n4", "n5", "n6", "n7",
     "n8", "n9", "n10", "n11", "n12", "n13", 
     "n14", "n15", "n16", "n17", "n18", "n19",
     "n20", "n21", "n22", "n23", "n24", "n25"]]
  processed_features = selected_features.copy()
  
  # Создает синтетическую переменную.
  processed_features["n20/n19"] = (
    magnet_data["n20"] /
    magnet_data["n19"])
  
  return processed_features

"""
Создадим функцию для выходящей переменной [25].

[25]: https://developers.google.com/machine-learning/glossary/#label
"""

def preprocess_targets(magnet_data):
  
  """Подготавливает выходящую переменную.

  Аргументы:
    magnet_data: исходный набор данных, тип: 'DataFrame'
    
  Возвращает:
    Выходящую переменную, тип: 'DataFrame'
  """
  
  output_targets = pd.DataFrame()
  output_targets["percent"] = magnet_data["percent"]
  
  return output_targets

"""
Разделим исходный набор данных на обучающее [26] и валидационное [27] подмножества.
В следствие его небольшого размеры, мы не будем создавать контрольное [28] подмножество, 
сохранив входящие переменные для обучения модели.

[26]: https://developers.google.com/machine-learning/glossary/#training_set
[27]: https://developers.google.com/machine-learning/glossary/#validation_set
[28]: https://developers.google.com/machine-learning/glossary/#test_set
"""

training_examples = preprocess_features(magnet_data.head(84))                    # Входящие переменные обучающего подмножества.
training_targets = preprocess_targets(magnet_data.head(84))                      # Выходящая переменная обучающего подмножества.

validation_examples = preprocess_features(magnet_data.tail(36))                  # Входящие переменные валидационного подмножества.
validation_targets = preprocess_targets(magnet_data.tail(36))                    # Выходящая переменная валидационного подмножества.

"""
Убедимся, что подмножества были сформированы верно, ознакомившись с  
их основными статистическими характеристиками.
"""

print("Обучающее подмножество — входящие переменные:")
display.display(training_examples.describe())
print("Обучающее подмножество — выходящая переменная:")
display.display(training_targets.describe())

print("Валидационное подмножество — входящие переменные:")
display.display(validation_examples.describe())
print("Валидационное подмножество — выходящая переменная:")
display.display(validation_targets.describe())

"""
Создадим поля [29], указывающие на тип данных.
Примечание: В Google для обозначения полей используется более интуитивный 
термин "feature column" [30].

[29]: https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
[30]: https://developers.google.com/machine-learning/glossary/#feature_column
"""

def construct_feature_columns(input_features):
  
  """Настраивает поля для TensorFlow.

  Аргументы:
  
    input_features: названия используемых входящих переменных.
    
  Возвращает:
    Множество полей TensorFlow, тип: 'set'
  """
  
  return set([tf.feature_column.numeric_column(my_feature)                       # 'numeric' указывает на то, что мы используем нумерические входящие переменные.
              for my_feature in input_features])

"""Приложение 1: Использование категорических переменных для обучения модели."""

"""
Чтобы использовать категорические переменные в моделировании, мы прибегаем к 
унитарному кодированию [1.1]:

Мы создаем словарь, содержащий все категорические значения, которые 
могут принимать интересующие нас переменные. Затем мы создаем 
*k*-мерный вектор (где *k* — длина созданного словаря) для каждого 
интересующего нас объекта в наборе данных. Эти векторы будут использованы 
в качестве входящих переменных. Если класс каждого объекта достоверно известен, 
векторы будут иметь бинарный вид, а единственным ненулевым элементом каждого 
из них будет индикатор [1] в соответствующей категории позиции. Если 
присутствует неопределенность, элементами вектора будут вероятности 
принадлежности объекта к той или иной категории.

Другим путем реализации унитарного кодирования является использование 
sklearn.preprocessing.OneHotEncoder [1.2].

[1.1]: https://developers.google.com/machine-learning/glossary/#one-hot_encoding
[1.2]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
"""

"""Шаг 3. Выбор алгоритма, обучение и оценка качества модели."""

"""

Мы используем высокоуровневый интерфейс линейной регрессии TensorFlow [31] и 
алгоритм FTLR [32] (разновидность градиентной оптимизации [33]).

Этот алгоритм хорошо подходит для многомерной линейной регрессии, по-разному 
масштабируя темп обучения [34] для разных коэффициентов (параметров) [35] модели. 
Подобная вариантивность оказывается полезной, если часть входящих переменных редко 
принимает ненулевые значения, а также хорошо поддерживает L1-регуляризацию [36].

Создадим функцию для обучения модели, использовав в ней FtlrOptimizer [37] и 
введя ограничения, предотвращающие взрыв градиента [38].

[31]: https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor
[32]: https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
[33]: https://developers.google.com/machine-learning/glossary/#gradient
[34]: https://developers.google.com/machine-learning/glossary/#learning_rate
[35]: https://developers.google.com/machine-learning/glossary/#weight
[36]: https://developers.google.com/machine-learning/crash-course/glossary#L1_regularization
[37]: https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer
[38]: http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L15%20Exploding%20and%20Vanishing%20Gradients.pdf
"""

def train_model(
    learning_rate,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Обучает модель линейной регрессии.
  
  Дополнительно: выводит информацию о процессе обучения,
  строит график функции потерь для обучающего и валидационных подмножеств.
  
  Аргумент:
    learning_rate: темп обучения, тип: 'float'
    steps: число тренировочных шагов (шаг = оценка одной корзины данных в двух
    направлениях, прямом и обратном), тип: 'int' > 0
    feature_columns: множество полей TensorFlow, тип: 'set'
    training_examples: столбцы тренировочного подмножества, 
    которые мы будем использовать в качестве входящих переменных, тип: 'DataFrame'
    training_targets: столбец тренировочного подмножества,
    который мы будем использовать в качестве выходящей переменной,  тип: 'DataFrame'
    validation_examples: столбцы валидационного подмножества, 
    которые мы будем использовать в качестве входящих переменных,  тип: 'DataFrame'
    validation_targets: столбец валидационного подмножества,
    который мы будем использовать в качестве выходящей переменной,  тип: 'DataFrame'
      
  Возвращает:
    Объект `LinearRegressor`, обученный на тренировочном подмножестве.
  """

  periods = 10                                                                   # Задает число отчетов о потерях, которые будут напечатаны.
  steps_per_period = steps / periods                                             

  # Создает объект LinearRegressor.
  my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)             # Настраивает FRTL-оптимизатор.
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)  # Предотвращает взрыв градиента.
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )
  
  training_input_fn = lambda: my_input_fn(training_examples,                     # Мы передаем созданную нами входную функцию в качестве анонимной,
                                          training_targets["percent"],           # чтобы иметь возможность указать в ней входящие и выходящие переменные,
                                          batch_size=batch_size)                 # а также размер корзин.
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["percent"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["percent"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Обучает модель, делая это внутри петли, чтобы мы имели возможность
  # периодически получать данные о потерях.
  print ("Идет обучение модели...")
  print ("RMSE:")                                                                # RMSE - англоязычная аббревиатура, обозначающая квадратный корень среднеквадратичной ошибки.
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    
    # Тренирует модель, начиная с априорного состояния.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Прерывается, чтобы сделать предсказания.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Рассчитывает потери для обучающего и валидационного подмножеств.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    
    # Периодически выводит данные о текущих потерях..
    print ("RMSE обучающего подмножества,  период %02d : %0.2f" % (period, training_root_mean_squared_error))
    print ("RMSE валидационного подмножества,  период %02d : %0.2f" % (period, validation_root_mean_squared_error))
    
    # Добавляет метрики потерь текущего периода в общий лист.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print ("Обучение модели завершено.")

  
  # Строит график потерь по отношению к периодам.
  plt.ylabel("RMSE")
  plt.xlabel("Периоды")
  plt.title("Корень среднеквадратичной ошибки RMSE для разных периодов")
  plt.tight_layout()
  plt.plot(training_rmse, label="обучающее подмножество")
  plt.plot(validation_rmse, label="валидационное подмножество")
  plt.legend()

  return linear_regressor

"""
Наконец, проведем обучение модели, найдя сочетание гиперпараметров [39], 
при котором завершается процесс конвергенции [40], а значение RMSE [41] становится 
достаточно малым, чтобы предположить, что оно приблизилось к величине неустранимой 
ошибки линейной регрессии [42].

[39]: https://developers.google.com/machine-learning/glossary/#hyperparameter
[40]: https://developers.google.com/machine-learning/glossary/#convergence
[41]: https://developers.google.com/machine-learning/crash-course/glossary#RMSE
[42]: http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf (страница 19, формула 2.3)
"""

#Случайный характер проводимой на этапе подготовки реиндексации приводит к тому,
#что при каждом новом выполнении кода данные из набора будут по-разному распределяться
#между обучающим и валидационными подмножествами. Подобный сценарий не дает возможности 
#универсально определить значения гиперпараметров. При написании этого текста, конвергенция 
#и значение метрики ошибки RMSE = 0.16 для валидационного подмножества достигались при использовании
#learning_rate = 0.000001, steps=500, batch_size=50.

_ = train_model(                                                                 
    learning_rate=0.000001,                                                      # Темп обучения
    steps=500,                                                                   # Число шагов.
    batch_size=50,                                                               # Размер корзин.
    feature_columns=construct_feature_columns(training_examples),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

