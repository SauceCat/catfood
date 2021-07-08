[Practical AI – Episode #117](https://changelog.com/practicalai/117)

# Getting in the Flow with Snorkel AI

featuring **Braden Hancock**



> **Chris Benson:** Was there a particular itch that you were scratching in that context, that actually led to Snorkel AI?

**Braden Hancock:** 

Very early on in my degree we were looking at what is the effective bottleneck for new machine learning applications; what is it that stops people from solving their problems as quickly as they'd like to? **The realization came that that bottleneck is almost always the training data.**

Deep learning was blossoming right about then, we saw these super-powerful models, feature engineerings becoming a lot less necessary; a lot of that can be learned now. But with the one caveat of "You can do all this if you have just mountains of perfectly-labeled, clean training data, ready to go for your specific task", and that in reality never exists. 

In Academia, it's "Download the dataset and then do something cool with it." But in industry, it's to prepare the data. Steps 1 through 9 is "Where am I gonna get my data? Do I have enough of it? Is it clean enough? These annotators are doing the exact wrong thing. I can clarify the instructions... Is this good now?" **It's iterating, and 80% of the work is making that training set.** After that, pulling off some state-of-the-art model in the open source and running is the easy part.



> **Chris Benson:** Tell us a bit about Snorkel AI, how the whole thing got started?

**Braden Hancock:**  

In the beginning it was like "This is an interesting idea. Let's run a quick experiment, pull up a Jupyter Notebook, test some of these ideas." Then it really seemed to work, so then it became a workshop paper, and then a full paper, and eventually a best-of paper, and an open source project, and then an open source ecosystem, and other derivative projects, and lots of collaborations.

We helped a few different organizations make industry-scale versions of this internally to really prove out the concept. A paper with Google, for example, that we were able to publish. And by the time that we were at the end of our degrees it was clear that there was just such a dramatic pull for this. The ideas were very well validated at that point, over probably 35 different peer-reviewed publications, but maybe more importantly, a whole bunch of different organizations that independently had seen success with these approaches.

**We just learned so much through that time about what you would really need to take this proof of concept and make it something that could be repeatable and with a relatively low barrier to entry,**  that doesn't require a room full of Stanford PhDs to make it successful. And that's part of what motivated the company, is the chance to now make this a fully supported, enterprise-ready and able to be shared with a whole bunch of different industries and company sizes, and in different work areas.



> **Chris Benson:** Before you dive into the specifics of the product and service offerings, could you talk a little bit about what you did learn?

**Braden Hancock:** 

**I think one of the big ones was interfaces.** As a grad student-supported open source project, you don't have a lot of time to polish up the front-end for people, so it's in the form of a Python package. One thing we did realize is, we were writing a lot of the same code over and over again. Soon we get lost between the forest of scripts and notebooks. **Whereas if you can set up a properly-structured interface and GUI, as well as other access points, you can really dramatically improve the likelihood of success.** 

If I was grouping it into other areas, **I'd say there's also infrastructure.** As a company, if you're going to depend on a piece of software, you need it to have certain things. Basic security, and logging, encryption, and compatibility with the data formats that you care about, and dependency management, parallelization, all these things.

I'd say another big piece of this is, as an academic you test often these ablations, you'll test a very specific problem, and "Can the model learn what I need it to?" **But in the wild, you often have actually just a problem you need to solve, and you don't necessarily care how that's solved; you just want a high-quality system.** And so you don't just have this one model that's ready to go with the data that you care about, that has an output that is exactly what you care about. **It's a pipeline. You've got preprocessing steps, you've got business logic, you're chaining together multiple models, or multiple operators. Some heuristic and some are machine learning-based.**

So this actually gets at one of the big differences, I'd say, in terms of fundamental value out of the Snorkel open source, versus Snorkel Flow, the business product. The latter is much more focused on building AI applications. **An application that solves your problem from end-to-end, rather than just a point solution for a part of the pipeline that is making a training set or training a single model.**



> **Chris Benson:** Could you now define what each of those are and describe what the differences between Snorkel open source and Snorkel Flow?

**Braden Hancock:** 

If you go to **Snorkel.org**, that's the website for the open source project and served as sort of our testing ground and proof of concept area for a lot of these ideas around "**Can we basically change the interface to machine learning to be around programmatically creating and managing and iterating on training sets?**" So that's what that is. It's pip-installable, you can pull it down now; it's got 4,000-something stars, and is used in a bunch of different projects.

**Snorkel Flow** is the offering: it's the primary product of Snorkel AI. It's based on and powered by that Snorkel open source technology, but then it just sort of expands to much more. It is now a platform, not a library; it comes with some of those **infrastructure improvements** that I mentioned before. It also bakes in a whole lot of the **intuitions** that we gained from the years of using the open source. There are certain ways that you can **guide the process in a systematic way** to creating these programmatic training sets or improving them systematically, really completing the loop, so that at every stage of the way you have some sort of hint at "What should I focus on next to improve the quality of my model, or of my application?"

So that platform, Snorkel Flow, is meant to be this much broader solution for supporting the end-to-end pipelines, not just the data labeling part, baking in a bunch of these best practices, tips and tricks that we learned over the years, of essentially writing the textbook on this new interface to machine learning. And it includes also some of those interfaces, like an integrated notebook environment for when you do want to do very low-level custom, one-off stuff... But also some much higher-level interfaces, like those templates I mentioned for labeling functions.

There are a number of ways where it can be a truly no-code or very low code environment for subject matter experts who don't necessarily know how to whip out the Python and solve a problem, but do have a lot of knowledge that's relevant to solving a problem.



> **Chris Benson:** If you could kind of give us a sense of what the open source side experience is like, what the benefit of the libraries are, that'd be fantastic.

**Braden Hancock:** 

One very simple example, one that we actually rely on in our primary tutorial just because it's very interpretable and almost everyone has the domain expertise necessary for it is training a document classifier. In this case, we could say the document will be emails, and **you wanna classify these as spam or not spam**.

One way you could do this in a traditional machine learning setting is get a whole bunch of emails that are sort of raw and unlabeled, look at them one by one and label them as "This one's spam, this one's not spam, that one's spam", and eventually you'll have thousands, or tens of thousands, or hundreds of thousands of emails that you need, to train some very powerful deep learning model to do a great job.

But when you do this process, if you'd ever tried to label a dataset, **you do find that very quickly there start to be certain things that you rely on to be efficient, or that are basically the science to you for why you should label things a certain way**. An easy example here might be lots of spam emails try and sell you prescription drugs. So you may see the word "Vicodin" in an email, and that's pretty clear to you this is not a valid business email, this is spam, and you can mark it as such. And you might eventually label over 100 emails that have the word Vicodin, and all of them are spam, for approximately that same reason, among other things. There's other content in the email, but that's what tipped you off. So if you could instead just one time say "And if you see the word 'Vicodin' in the email, good chance that this is more likely to be spam, rather than (we'll call it) ham, or not-spam."

You could write that, apply that to hundreds of thousands of unlabeled data points, and get in one fell swoop hundreds of labeled examples. And those labels may not be perfect; there may actually be a couple examples in there, some small portion where it actually was valid; someone was asking "Did you see where my Vicodin was put?" I'm not sure. I won't guess.

But basically, **these noisier sources of supervision can be then much more scalable, much faster to execute, easier to version control and iterate on than individual labels are**... And if you can layer a number of these on top of each other, and basically then let their votes be aggregated by an algorithm, one that we developed at Stanford, you now have the ability to get - maybe not 100 perfect labels, but 100,000 pretty good labels, and it takes about the same amount of time. And as we've seen time and time again in recent years, the size of the dataset seems to keep winning the day when it comes to getting high-performance with these models.

**It essentially is a way of building and managing training sets very quickly, often at a much higher rate of production, as well as just much larger magnitude.**



> **Chris Benson:** What is a typical scenario that you're finding with customers, where they do need to level up? What is it that they are now facing, that is a clear step-up and they need the enterprise approach at this point?

**Braden Hancock:** 

**One of the big ones is just the guidance**. I think with the proof of concept library, the open source, over the years of using it, we knew what to look for; how accurate is accurate enough for a labeling function, how many do I need, how should I come up with ideas for what a valid labeling function could be, how could I integrate external resources that I may have, like a legacy model that I wanna improve on, or maybe an ontology that belongs to the business, that has information in, and how should I integrate that.

So there's a lot of what would otherwise be **folk knowledge** if you're using the open source that you just only get through experience, that we've been able to really **bake in and support in a native, first-class, guided way in the platform**, and that's a big difference-maker for a lot of people.

When you have not just a one-off project, when your goal is not just to fill a table in a paper, but really to build something that you have confidence in, that you can come back to, that you can point to in the case of an auditor, and whatnot... **There's extra value in managing all these different artifacts.**You've got often many applications that you care about and many teams working on it, many different artifacts that you create, whether that's models, or training sets, or sets of labeling functions... **So there is an element here that's as well just the data management side of things, and tracking and versioning and supporting all of those types of workflows.**

On the modeling side, that is entirely unique to the platform with respect to the open source. **We have a bunch of industry-standard modeling libraries integrated with the platform**, so if you do want to train a scikit-learn model - sure; or some of the Hugging Face transformers right there. Flare is another one. XGBoost. So a lot of these libraries we've kind of unified behind a simple interface, so that it can be a sort of push-button experience to try out a number of different things, and hyperparameter tune, and whatnot... But with the goal really being of -- **you'll find most of the time you'll get the biggest lift by actually improving the training set rather than the model.**

**I guess that actually moves us on to the fourth part, which is analysis.** We have a whole separate page with a bunch of different components that effectively take a look at how your model is currently performing, and where it's making mistakes, and why it might be making those mistakes, and then makes concrete recommendations for what to do next.

In some cases it's "Yes, actually your training set looks pretty good. The learned labels that we're coming up with actually line up pretty well with ground truth. So if you're making mistakes here, it's probably because -- it's your model now, so you need to try a more powerful model, or hyperparameter tune a little bit differently. I think that's where a lot of machine learning practitioners naturally go, immediately to the model and hyperparameter tuning, **when in reality almost always the far larger error bucket is there are whole swathes of your evaluation set that have no training set examples that look at all like them**.

There are basically just blind spots that your model has, and now in the platform you can go ahead and click on that error bucket, go look at those 20, or 100, or however many examples where none of your labeling functions are applying, so this is not reflected at all in your training set, and **write some new supervision that will add effectively examples of that type to your training set**, so that the next model you train will know something about those types of examples, and can improve upon them.



> **Chris Benson:** What is that special sauce, to some degree, that you guys were really looking to introduce into the marketplace with this platform?

**Braden Hancock:** 

I think what really moves the needle is the fact that with this approach and with this platform, **machine learning becomes just more practical, more systematic, more iterative**. We've seen these used successfully, and we'll continue to build out the areas for applying this to other modalities as well... But this paradigm is really agnostic to the data modality and most problem types. **At its heart, it is a machine learning problem where you have a training set and you have a model, and when your model is making mistakes, it's often due to what is or isn't reflected clearly enough in your training set.**

So for any of these problems, there are different types of labeling functions that you write for a classification problem versus an extraction problem, or whatnot... But fundamentally, once you scrape off that top layer, it looks very similar. So this platform really is meant to solve a wide variety of problem types, and work in a whole bunch of different industries and verticals and whatnot... **Because again, under the hood, they're all relying on the same, basic, fundamental principles about how machine learning works.** And it was with that in mind that we built the platform.



> **Chris Benson:** Could you describe a little bit about how you might integrate in with other tools that are widely used within this industry? What kind of integrations do you have, and how that really helps the practitioner get through the process of modeling that they're trying to do?

**Braden Hancock:** 

One of the things that we learned from the open source project was **the importance of having intuitive, natural, modular interfaces to different parts of this pipeline**. The labeling functions as well, the models, all that. So we kept that design principle very much in mind as we designed the platform, and we've made sure that every step of the pipeline can be done either in the GUI, or via an SDK that we provide.

So that means that you can write labeling functions via these nice GUI builders that we've got, or you can define completely arbitrary black box labeling functions via code in the notebook, push those up, and then they're treated the same way in the platform. Same thing with the training sets; you can create a training set and then go to the models page and identify the model that you want, set up your hyperparameters and train it there with a button, or you can use the SDK to export your training set, training your own model, and then just re-register the predictions, push them back up, just some very lightweight, assign certain labels, and then use the analysis page to still guide you.



> **Chris Benson:** When you guys are getting together and hanging out and talking about what-ifs, what are some of those what-ifs that you're willing to share?

**Braden Hancock:** 

As I mentioned a couple times, **there are different modalities to consider**, and the way that you write labeling functions over images is fundamentally different than the way that you write labeling functions for text. So just given where the market pull was initially, we've started focusing on text, but we absolutely plan to bring in some of that other research we've done as time goes on, over the coming months and years.

I'd say in addition, another area that's really interesting to us, so where we would have this unique leg up based on the approach that we're taking, is **the monitoring side of things**. When you acknowledge that most applications are gonna go deploy, it's not "Great! I've got my model now. Deploy it, and set it and forget it." **Test distributions change, the world shifts.** People talk about different topics; different words get different meanings. Covid was not a part of the discussion a year ago, and now it's a huge part of the societal fabric of what gets talked about on social media.

So the fact that you do very frequently need to iterate on your models, improve them, as well as you'd like to know preferably more than just a single number - the accuracy of my model, is that going up or down? It's really interesting to see what types of examples am I starting to get more right or more wrong? What subsets of my data are diverging, basically, from what they were when I was trained? What's really interesting is after you've written these labeling functions, there are essentially a whole bunch of different hooks into your dataset.

They each observe different slices of your data that have different common properties, and these could effectively become monitoring tools for you, because **you can now observe how those labeling functions increase or decrease in coverage over time when applied on the new data that's streaming through your application**, and inform you when -- you could basically set up automated alerts showing you "Now is the time to go and update things" or "Here's some suspicious activity going on", based not just on "Did the number go up or down?", but "We're seeing movement in different parts of the space where your model is operating. Take a look."

That maybe appeals more to the technical/nerdy side of things, but I think it's a really interesting problem, one where you've got that information. You have already identified for you these very interesting angles on your problem, and so why not use those to help **guide the post-deployment life of a model**.



> **Chris Benson:** What are some of the really interesting things that you've seen customers doing with this?Particularly things that were outside of what you might have expected. 

**Braden Hancock:** 

Two things that I've found personally very cool - **one of them is the privacy preservation aspect of this approach.** That was not necessarily a top priority or top-of-mind when we were developing these techniques at Stanford. But it's been really cool to see different companies that have the very desirable goal of "**We'd like to have our data being seen by fewer humans.**" We'd like to have fewer people reading your emails, fewer people seeing your medical or financial records; how can we do that while not sacrificing the quality of our machine learning models?" So it's been really interesting to see them, and working with them, coming up with these setups where **now they can take a very small, approved subset of the data to give them ideas for how to write labeling functions**, or label a test set to give them a sense of overall how is quality.

But then the vast majority of their data never gets seen by a human now. They can take these programs they've developed to go label those automatically, use them to train a model, and then get back just the final weights of the model. It's really neat to see, and I'd love to see that thread continue... Not just from a privacy preservation standpoint, but also - **we keep seeing articles about the PTSD that you get as annotator over these awful domains.** 

I think another interesting application we've seen was we had one customer who had an application that was affected by Covid. And they came to us and said "Okay, this was not part of our scoped work, but this suddenly matters a lot to us, and our typical process would take about a month. Do you think you can help us? **Could we try and use Snorkel to get some result faster?"**

And since it was very early on and we hadn't necessarily had a lot of time to train using the platform, we said "Sure, we've got some ideas. Give us a sec." We threw three of us in a war room for the day, ordered some burritos and hacked away, and by the end of the day we were able to extract the terms that they needed with over 99% accuracy on their application. That was achievable with a model that was trained on tens of thousands of examples, which we didn't need to label. We were able to quickly come up with **"What are the generalizable rules or principles here that we could use to create a training set to train a model that now can handle edge case and things much better than these rules?"** and get then get the high quality that they needed. So that sort of live action, the nerds save the day kind of moment...



> **Chris Benson:** Do you have any insight or any thoughts into where we're going in terms of us moving along this curve from these static-label datasets that we were talking about, historically, at the beginning of the conversation, to this dynamic, especially since Covid has struck, the ever-changing world on a day-to-day basis - what's that trajectory look like, and how are you guys preparing for that?

**Braden Hancock:** 

Because people are aware now of some of the costs that come with machine learning. The promise of machine learning is very much being broadcast, how it's the future and it solves a lot of problems, but there do end up being these very practical -- I won't say necessarily limitations, but gotchas, or costs really, that you need to be aware of. I see this reflected a little bit in the way that companies are starting to prioritize more the ability to see **"Where did my model learn this?" That auditibility**...

So they realize that that's important in a way that they maybe didn't before. I think they're also realizing just from an economic standpoint that **training data is not a one-time cost. This is a capital expenditure, this is an asset that loses value; there's a half-life to these things**. So you start seeing these ongoing, regular, cyclical budgets to get the training data, even just for a single application -- not "We need more data to train more models for more applications, but to **keep this application fresh and alive**."

It's super-interesting, and it changes the way that you account for the cost of different applications you might use, because there's a certain way that you maintain imperative (we'll call it) Software 1.0, and there's a different way that you maintain this machine learning-based Software 2.0 way of solving a problem. It's something people are learning, and I think that's all an interesting part of the conversations that we're having with different customers as **they realize how this can maybe change the way that they approach their machine learning stack in general**.



> **Chris Benson:** Okay. As we wind up here, I'm finally getting to ask what is always my favorite question anytime we're getting to talk to someone such as yourself... Blank sheet of paper, what are you excited about right now in the space of machine learning and AI? What is the thing that has captured your imagination, whether it's work-related or whether it's not work-related? Something cool out there... What's got you going "That's the thing I'm really interested in tracking, either on my own, or through the company, or whatever?" What's cool?

**Braden Hancock:** 

So many things... I'd say there are a number of areas that are super-important; super-hard, but super-important. And I'm glad to see that they're getting the attention that they deserve, or at least that we're trending in the right direction. **And that stems around the privacy, the fairness, the bias...** A lot of that I think is just super-hard. If anyone says that they've got a solution to that problem, I'd be very dubious... But I think we are marching toward progress there, and that's something that I'm certainly gonna watch with great interest, and hope that we can be a part of the solution there. That's one piece.

What I think may be a little closer to my personal research agenda in history - a lot of that's centered around **how you get signal from a person into a machine**. So a lot of my research through the years has been seeing how high up the stack can we go. There's this figure in my dissertation that compares basically the computer programming stack to the machine learning stack.

Computer programming -- computers run on these ones and zeros; they run on individual bytes and bits, but nobody writes ones and zeros code. We write in higher-level interfaces, like a C, or even like a SQL or something, that compile down sometimes multiple times into this low-level code that you're then gonna actually run on. And I'd say similarly, machine learning runs on individual-labeled examples; that's how we train it, that's how we express information to it. **But it feels fairly naive, actually, to one-by-one write these ones and zeroes, write these trues and falses on our individual examples.**

So I think that there's a lot of really interesting things that can be done around higher-level interfaces of **expressing expertise** that then in various automated or just sort of assisted ways can eventually result in the training sets that have the properties you need to actually communicate with your model, use the compiler, essentially, the optimization algorithm that's in place to transfer that information.
