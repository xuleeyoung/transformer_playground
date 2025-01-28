# Transformer Playground
A transformer and a mini-GPT implementation under PyTorch framework.

### Train the model to learn multiplication. 
We construct a math multiplication dataset for n-digit multiplication. Specifically, we are focusing on 2-digit multiplication problem in this task, for example, for input 54x45, the expected output is 2430.
![loss](multiplication_loss.png)


### Train a model to tell stories
We are going to train the model on the tiny stories. Please open the link, download the TinyStories-train.txt file, and put it in your data folder like below.
```
HW4 /
--| transformer.ipynb
--| data /
----| TinyStories-train.txt
----| ...
```

In early iterations, the model will not generate anything meaningful, but as you train the model for more iterations, you will gradually see some interesting stories. The stories can be imperfect because we are only running a small model, but it should generate real words mostly, and follow grammar rules to some extent. Feel free to run it for longer (increase max_iters) or make the model size larger if you have time and want better results, but the default setting should be good enough.

_Here are some generated texts during training_:

**1. Iteration 501/15000: training loss 1.7922**
- Story (1): 
Once upon a time hin bom the motind, "I walle pat a and it, be a look too cher an to and the soul. They saw saw the their to proce to the ime to soma. She ad and trie it to boulte. She the a days to the whing so pok it. She they hownthin's to her. He der was was terir sarked tlit bootat ithat twe at fun it and ware forended she outher. Her did ad sallled and Tis the comeltele, "Doved buron, "Lily't, her she cacyir to but flig therat a shir shomed thid. The art bane, anve bry happy her was shey with belit fins flird thens in alllings theim in now toge soons. "Wan, hey as at goo therar, I downe aver the ats and and wat they cad bill theis, she sploot to tich becat igh thor they. He domm they dad bon the she said and shie. She aned the sort as in it then togart.

- Story (2): 
Once there was an take. It, "Why bolelle we the to pry!"

- Story (3): 
One day, he heave a she clooooth her butle the pil bear the cat shellle. They we tho the they felt. It was shey bich is the wie fat fors fil. She he hear hey had lay shee betig the walfoll she and and he fof and.
Her she saw to sitede with was to the go theirtth and hage bot ir sthe tore.
Lily "Lily, "Lely wis's, he shit hit tat ir her an alll skeated. 
Bunly here want souck and Somalld the the touet.
They scared as aid "Tim sthe was and the bor icith."
She wim bigl lovens it but it and "Lut to am the sant longath Thim, "Lou," Then dome lanet and big, Sheche and his and live an saw the that bit to hes atingede. They the dot claver they. The my sared tid sogld out the hel the gigar in. They litle wile. They hey saw sot diggl. 
She smaid, The s at and santes Sane. Hent sad shirrr, "Yeam. "Le lookled to the sto cat tat welke to take boy."
Lily dampp werees tollt frated. They loty they wigh all agro brom.
The helt bill there arake! He they was hitan ifomm. "I dou d and hellp tooo the ipp to tire stometh int in then brat

**Iteration 7501/15000: training loss 0.7781**
- Story (1): 
Once upon a time, there was a boy named Timmy. Buzz was hungry. He had a big smile. He wanted to see the blocks in the park.
"Can I touch it?" Timmy asked. He went to the park.
Timmy was full of blocks. He wanted his block. He asked his mom and ran to the park. He saw some blocks on the paths.
"Wow, Timmy!" Lily shouted. The paitents looked for it!
"Sure, he did and sang and glowed. The bag is barking and closely."
Timmy was hungry and scared. He did not cross his bike. He saw the painter. He saw two bread animals. He left them and put it back in his towards back. He saw what his fhoot was lying, and he swung to Timmy and tried to cross. At them, Timmy were scared and sorry.
They crashed their hands and flew away. They did not know of him.
Timmy didn't like that. He swang and thought. He washed his finger with their doll. He was happy. He was friends again, and they did not give up.

- Story (2): 
Once there was an old lady named Lily. She loved to clean up her dog when she met accept love. One day, Lily went to the park. She saw a slide there. She did not know where her friend was crying. She ran to them, but they did not want a superhero. They stuck it on the pit.
Mom heard a noise and came to the park and saw them in small. She was scared and sad. She asked Sarah and Ben in her room.
"Why did you stuck?" Lila said.
She ran to the kitchen and came home. They saw their mom and dad. "This is different and dangerous. Do you like to make your chicken?" Sara
Sarah hugged her and said, "We will like spots, patterns. And we sweet swan between the bridges." They walked to the kitchen and saw the old lady. They walked to her mom. They saw Sarah feet holding.
They spent the days and played together. They felt scared of their white secret sandwiches and made a flag. They did not want to hurt the man again. They did not want to share anymore.
They walked to their mom. She picked up the secret and hugged her and said the made the 
- Story (3): 
One day, they found a lot of shapes and a round bird paw.
"Let's touch it," Ben said. "But shapes is too late. What do you told it to his?"
"Yies, you did a girl!" Ben said. He low scared the bird again.
They saw birds, blue, birds. They did not want to try it. They wanted to go back.
"Thank you, Mom?" Tim cried. "You were brave. They were being kind and having fun. You are twins. What do you want to display?"
Mom looked up. She saw the bird and smiled, and said, "I love candy. But you have to clean up my ball. It is not harm. Good to think tomato. It has a beautiful candy in the distant stead. You cant stories about it. That way, I have a new friend. It is a great thing. It is a problem."
Tom hugged Lily and clapped again. He thanked the candy and went to the distant. He said, holding his new plate with the candy, and they splashed and together. They made the pillow with the paper and the book together.
Their mom gave him the spaghetti and said, "I will not listen to you. I have any plates and shatter to all them a


**Iteration 15000/15000: training loss 0.6190**
- Story (1): 
Once upon a time, there was a little bird named Tweety. Tweety loved to play with his rock with all kinds of toys. One day, he found a crack that made him go vroom. They were very happy to see the crack hiding some pictures.
The crack bought the story and checkered her eyes.
Tweety ran to the city and said to the crack. But it would not come true. Sam sahed the rick to think where he would find it. Mr. Smith looked around and saw many different things to do. He had no finguing to come inside next to the forest.
At the end of the dark, Lily found a smalll hole in the window. She was so happy to have found the honey for a long time ago. They said, "Let's go inside, Mom!" They hugged Lily and said, "Thank you for taught us to be kind. And take the honey soon soon." From then on, Lily practiced pupping and hit hands on the walllet. They had so much fun with their new honey that took Lily honked in dough.

- Story (2): 
Once there was an old man. He wanted to back to his truck. He could not wait to try it.
"Let's go, Tom," his mom replied. "Maybe we can have something else too."
"OK, Ben," Tom said. "We can make something big and beautiful."
They worked together to display the truck. Lily was happy to show them how to fix them them. Her favorite toys mixed so they could show them alll her friends. They were happy and gentle with their friends.

- Story (3): 
One day, Tim was playing with his friend, a little fish, came and saw a big tree with a big rock, and said, "Help, what's inside?" Tim agreed, and Tim followed Tim for a long time. They scooped the rock and turned around the tree, packing from the tree. They were happy to be their friends.

