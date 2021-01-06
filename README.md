[Лаба](#1 лабораторная работа)
Тест что ридми
***
```
Тест кода
using MarkdownSharp;
using MarkdownSharp.Extensions.Mal;

Markdown mark = new Markdown();

// Short link for MAL - 
// http://myanimelist.net/people/413/Kitamura_Eri => mal://Kitamura_Eri
mark.AddExtension(new Articles()); 
mark.AddExtension(new Profile());

mark.Transform(text);
```
# [1 лабораторная работа](https://github.com/GorVad/MISIS-NN-ML/blob/master/CNN/CNN_Classification.py)
