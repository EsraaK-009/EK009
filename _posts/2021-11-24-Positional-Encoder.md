---
layout: post
title: "Positional Encoder"
description: "Positional Encoder"
date: 2021-11-24T07:00:00-07:00
tags: Arabic
#image: /img/foo/bar.png
---

## Positional Encoder

<p dir='rtl' align='right'>
مكان الكلمة في الجملة من الحاجات المهمة جدًا لينا عشان نفهم معناها و كمان بتأثر على القواعد فعشان كده محتاجين الموديل يحافظ عليها و يحطها في حساباته و هو بيترجم أو بيعمل أي عملية تانية.

طبعًا الـRNN كانت بتعمل كده بالفعل و بتحافظ على الـsequence، بس احنا اتخلينا عن الجزء ده لأنه كان عامل مشكلة بس مكان الكلمة لسه مهم، فمن هنا كان لازم نلاقي بديل و هو الـPositional encoder.
</p>

<p dir='rtl' align='right'>
  بداية، نفكر سوا أسهل حاجة ممكن نفكر فيها إن كل كلمة تكون برقم من أول 1 لحد أخر الجملة، لو الجملة 7 كلمات، أخر كلمة تأخد رقم 7 بكل بساطة، بس
 الأبسط بيحطنا في مشاكل كتير، ماذا لو دخل للموديل بتاعنا جملة أطول من الجمل اللي اتدرب عليها؟ ماذا لو الأرقام كانت كبيرة أوي و عملت لي مشاكل في الحسابات؟ لأ و هنا كمان أنا عندي مشكلة تانية بتسببها الأرقام الكبيرة و هي إنها بتبعد الكلمات عن بعضها جدًا في الـspace! فبيقلل الـposition similarity ما بينهم و بيضرب الدنيا
</p>

<p dir='rtl' align='right'>
فهنا بقى هنحط شوية قواعد عشان نقدر نقول إن ده encoding لمكان الكلمة و مناسب أقدر أدخله على حسابات الموديل من غير ما يبوظها:
1 - لازم يكون unique لكل كلمة.
2 -الموديل بتاعي يعرف يتعامل مع طول جملة هو حتى متعرضش له قبل كده.
3 - لازم يكون deterministic.
</p>

<p dir='rtl' align='right'>
الحل اللي قدمته Attention is all you need بسيط جدًا و في نفس الوقت مبهر،  باستخدام الـwave frequency بيعملوا encoding خاص لكل مكان في الجملة و فكرته ذكية جدًا عشان بيحقق كل الحاجات المطلوبة 
</p>

![Equation]({{ site.baseurl }}/images/Encoder/equation.png)
![Vector]({{ site.baseurl }}/images/Encoder/vector.png)

----
 
<h3 dir='rtl' align='right'>ليه الـsin و الـcos? </h3>

<p dir='rtl' align='right'>
احنا عاوزين حاجة تدينا رقم unique و في نفس الوقت مش كبير و له حدود، الـsin و الـcos بيكون الrange من 1 لـ -1 و كمان فكرة الترددات بتخليها تمسك أقل أقل تغير في المكان. كأنها زرار بيعلي الصوت بس مش discrete لأ، شبه الراديو القديم كل ما تعلي سِنة صغيرة بيتغير. 
</p>

<p dir='rtl' align='right'>
بس فيه مشكلة دلوقتي! احنا قولنا إن الـsin موجة! و ده معناه إنها بتدي نفس القيمة تاني كل شوية، طب ما كده مبقتش unique!
عشان يتغلبوا على النقطة دي بيكون الـfrequency قليل جدًا جدًا كل ما عدد الـindex بيزيد و عشان كده المعادلة في الصورة دي  بتعمل geometric progression. دي حاجة تانية بس باختصار كمثال عليها إننا بنمشي بتتابع بيقل مع كل خطوة. مثال: [ 1 , 1/2 , 1/4 , 1/8 , ...] 
</p>

![Waves]({{ site.baseurl }}/images/Encoder/index.png)


<p dir='rtl' align='right'>
في الصورة دي شكل لأربعة dim. بس، و لكن الـمتجه في العادي بيبقى أكبر من كده بكتير، فالصورة بتكون أعقد بكتير و فعلًا unique لكل position و بتبقى مترتبة في صورة matrix بحيث يقدر يلقط التغيرات
</p>

----

<h3 dir='rtl' align='right'>ليه بنقسم على 10000؟ </h3>


![10000]({{ site.baseurl }}/images/Encoder/why10000.png)

<p dir='rtl' align='right'>
في حاجة لطيفة تانية، في البيبر نفسها في السطر الموجود فيه المعادلة بتقول إنهم اختاروا المعادلة دي لسبب تاني إنها هينفع مع الـrelative position encoding بسبب اثبات ما هحطه اللينك، بس ده مهم ليه؟ عشان البيبر مكانتش معمولة كهدف للـtext بس، احنا لسه عندنا graphs و أنواع داتا مختلفة كتير. فهما اختاروها عشان كده كمان.
</p>

![why_eq]({{ site.baseurl }}/images/Encoder/why_eq.png)

----

<p dir='rtl' align='right'>
الكود باستخدام numpy: 
</p>



```python
def positional_encoding(pos,model_size):

  POS = np.zeros((1,model_size)) 
  for i in range(model_size):
    if i % 2 == 0: #even number
      POS[:,i] = np.sin(pos/10000 *(i / model_size)) 
    else:
      POS[:,i] = np.cos(pos/10000 *( (i-1) / model_size)) 
     
      
  return POS
```
----

<h3 dir='rtl' align='right'> مصادر:</h3> 

* <a>https://kazemnejad.com/blog/transformer_architecture_positional_encoding/</a>
* <a>https://www.youtube.com/watch?v=1biZfFLPRSY</a>
* <a>https://datascience.stackexchange.com/questions/82451/why-is-10000-used-as-the-denominator-in-positional-encodings-in-the-transformer</a>
* <a>https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model</a>
* <a>https://timodenk.com/blog/linear-relationships-in-the-transformers-positional-encoding/</a>
* <a>https://github.com/tensorflow/tensor2tensor/issues/1591</a>
* <a>https://medium.com/swlh/elegant-intuitions-behind-positional-encodings-dc48b4a4a5d1</a>
