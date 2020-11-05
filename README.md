# 使用方法

```python
>>> python model.py -h
optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        输入数据文件的位置
  -s S                  是否为诱导模型
  -o OUTPUT, --output OUTPUT
                        输出数据文件的位置
```
基础模型

```python
python model.py -i example/example1.xlsx
```
诱导模型
```python
python model.py -i example/example2.xlsx  -s yes

```
改变输出文件
```python
python model.py -i example/example2.xlsx  -s yes -o res.txt

```