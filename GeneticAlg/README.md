# Лабораторная работа 2. (GenAlg)

--------------------------

Реализован генетический алгоритм на `CPU` и `GPU`. 
Реализация включает в себя 2 пакета: `GA_CPU` и` GA_GPU`. Обе реализации используют одинаковую логику генетического алгоритма:
инициализация → оценка fitness → селекция → скрещивание → мутация.

Версия на `CUDA` была реализована с использованием библиотеки `CURAND` для генерации случайных чисел на девайсе. 
Для работы с массивами использовался модуль `Thrust`. Все шаги генетического алгоритма были реализованы на `GPU`. 
`CPU` используется для инициализации алгоритма, копировании лучшего результата `fitness` для принятия решения
о преждевременном завершении алгоритма и для возврата результата оптимизации.
Реализация включает в себя следующие ядра:
1. `setup_rng` - инициализация генераторов
2. `init_population_kernel` - инициализация начальной популяции особей случайными коэффициентами.
3. `fitness_kernel` - вычисление функции ошибки для каждой особи.
4. `crossover_kernel` - генерация нового поколения путем скрещивания.
5. `mutation_kernel` - Внесение случайных изменений в популяцию для поддержания разнообразия.
Селекция выполняется так же на `GPU`, только с использованием `Thrust`. 

Исходная функция:
$$
f(x) = 1 - 5x + 20x^2 - 4x^3 + 5x^4
$$
Каждая особь представляет собой набор коэффициентов полинома.

Результаты запуска алгоритмов с разными параметрами:

```
===== Parameters =====

Num Points |        Population Size |   Max Iterations |    Max const iterations |
1000                500                 2000                200


===== GA RESULT CPU =====
Best fitness: 5.67818e-06
Total algorithm time: 1112.22 ms
CPU time: 1112.19 ms
Last generation: 2000
Coefficients:
  c0 = 1
  c1 = -5
  c2 = 20
  c3 = -4
  c4 = 5


===== GA RESULT GPU =====
Best fitness: 3.14595e-05
Total algorithm time: 1379.28 ms
GPU time (without mem copy): 1265.82 ms
Last generation: 2000
Coefficients:
  c0 = 1
  c1 = -5
  c2 = 20
  c3 = -4
  c4 = 5


===== SPEEDUP =====
Total speedup (CPU total / GPU total): 0.806x
Compute speedup (CPU time / GPU time): 0.879x
```

```
===== Parameters =====

Num Points |        Population Size |   Max Iterations |    Max const iterations |
1000                1000                2000                200


===== GA RESULT CPU =====
Best fitness: 5.07631e-06
Total algorithm time: 2151.65 ms
CPU time: 2151.6 ms
Last generation: 2000
Coefficients:
c0 = 1
c1 = -5
c2 = 20
c3 = -4
c4 = 5


===== GA RESULT GPU =====
Best fitness: 6.00804e-06
Total algorithm time: 1386.68 ms
GPU time (without mem copy): 1277.69 ms
Last generation: 2000
Coefficients:
c0 = 1
c1 = -5
c2 = 20
c3 = -4
c4 = 5


===== SPEEDUP =====
Total speedup (CPU total / GPU total): 1.55x
Compute speedup (CPU time / GPU time): 1.68x
```

```
===== Parameters =====

Num Points |        Population Size |   Max Iterations |    Max const iterations |
1000                2000                2000                200


===== GA RESULT CPU =====
Best fitness: 5.76037e-07
Total algorithm time: 4277.57 ms
CPU time: 4277.5 ms
Last generation: 2000
Coefficients:
c0 = 1
c1 = -5
c2 = 20
c3 = -4
c4 = 5


===== GA RESULT GPU =====
Best fitness: 1.56118e-06
Total algorithm time: 1400.88 ms
GPU time (without mem copy): 1282.37 ms
Last generation: 2000
Coefficients:
c0 = 1
c1 = -5
c2 = 20
c3 = -4
c4 = 5


===== SPEEDUP =====
Total speedup (CPU total / GPU total): 3.05x
Compute speedup (CPU time / GPU time): 3.34x
```

По результатам работы алгоритма можно сделать вывод, что коэффициенты заданного полинома по выборке значений восстанавливаются корректно.

С увеличением размера популяции наблюдается рост ускорения GPU-реализации по сравнению с CPU. При небольших размерах задачи использование GPU нецелесообразно,
поскольку накладные расходы на инициализацию вычислений и передачу данных между CPU и GPU занимают значительную долю общего времени выполнения.