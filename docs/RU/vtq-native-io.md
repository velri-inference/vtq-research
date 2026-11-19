# Необязательный CPU-бэкенд для Python-писателя и загрузчика VTQ

`src.vtq_native.NativeVtqCodec` упаковывает уже найденные коды и распаковывает `W_rot` в FP32. Он **не** меняет Витерби, выбор формата, байтовую раскладку или ядро matvec в движке. Путь без `codec=` остаётся NumPy-эталоном. Публичный API принимает объект явно: импорт не вызывает компилятор, не скачивает код и не читает путь к библиотеке из квантованной модели.

Пример сборки на машине, на которой библиотека будет использоваться (проверки в этой серии — только на Colab G4):

```sh
c++ -O3 -std=c++17 -march=native -ffp-contract=off -fPIC -shared \
    src/native/vtq_pack_tiles.cpp src/native/vtq_unpack_tiles.cpp -o /tmp/libvtq_io.so
python -m src.test_vtq_native_io --library /tmp/libvtq_io.so
```

Для переносимой сборки не использовать `-march=native`; не использовать `-ffast-math`. Загружать только свою доверенную библиотеку.

```python
from pathlib import Path
from src.vtq_native import NativeVtqCodec
from src.vtq_format import VtqWriter, VtqFile

codec = NativeVtqCodec(Path('/tmp/libvtq_io.so'))
# При записи: VtqWriter(path, formats, codec=codec); add_matrix / finalize как обычно.
reader = VtqFile(path, codec=codec)
try:
    weights = reader.decode_matrix(prefix, device='cuda')
finally:
    reader.mm.close()
    reader.f.close()
```

Совместимость не требует нового заголовка файла. `test_vtq_native_io` проверяет обе перекрёстные комбинации writer/reader, неполные тайлы, все форматы в срезах одной матрицы, некорректные кодовые слова, обычный BF16-тензор и FP32 после обратного вращения. [Результат на G4](../../results/exp26_native_io_api.json): файлы совпали побайтно, результат FP32 совпал побитно на CPU и CUDA; на каждом пути пройдены 37 отрицательных проверок. Дополнительно [публичный `codec=` на реальных данных](../../results/exp26_public_api_samples.json) проверен на крупнейшей сохранённой матрице каждой из 8 штатных и 2 дополнительных моделей: во всех случаях по три пары загрузок дали одинаковые FP32-хеши. Это проверка интеграции на выборке, не замена полному exp26 по всем матрицам. Проверка `verify=True` в писателе намеренно сохраняет независимый NumPy-декодер. Экспериментальный TSB1 с набором gain-conditioned LUT не является обычным VTQ и не поддерживается этим API. Отчёты различают время упаковки, время полного экспорта, материализацию на GPU и инференс: [критерий на всех моделях](universal-gate.md).
