# Excavating AI

### The Politics of Images in Machine Learning Training Sets 

By Kate Crawford and Trevor Paglen  | Link: https://excavating.ai/



In this essay, we will explore why the automated interpretation of images is an inherently social and political project, rather than a purely technical one. Understanding the politics within AI systems matters more than ever, as they are quickly moving into the architecture of social institutions: deciding whom to interview for a job, which students are paying attention in class, which suspects to arrest, and much else.

For the last two years, we have been studying the underlying logic of how images are used to train AI systems to “see” the world. Methodologically, we could call this project an *archeology of datasets*: we have been digging through the material layers, cataloguing the principles and values by which something was constructed, and analyzing what normative patterns of life were assumed, supported, and reproduced. By excavating the construction of these training sets and their underlying structures, many unquestioned assumptions are revealed. These assumptions inform the way AI systems work—and fail—to this day.

## Training AI

Building AI systems requires data. Supervised machine-learning systems designed for object or facial recognition are trained on vast amounts of data contained within datasets made up of many discrete images. 

Training sets, are the foundation on which contemporary machine-learning systems are built. They are central to how AI systems recognize and interpret the world. These datasets shape the epistemic boundaries governing how AI systems operate, and thus are an essential part of understanding socially significant questions about AI.

But when we look at the training images widely used in computer-vision systems, we find a bedrock composed of shaky and skewed assumptions. The project of interpreting images is a profoundly complex and relational endeavor. Images are remarkably slippery things, laden with multiple potential meanings, irresolvable questions, and contradictions. Entire subfields of philosophy, art history, and media theory are dedicated to teasing out all the nuances of the unstable relationship between images and meanings.

Images do not describe themselves. We see images differently when we see how they’re labeled. The circuit between image, label, and referent is flexible and can be reconstructed in any number of ways to do different kinds of work. What’s more, those circuits can change over time as the cultural context of an image shifts, and can mean different things depending on who looks, and where they are located. Images are open to interpretation and reinterpretation.

This is part of the reason why the tasks of object recognition and classification are more complex than Minksy—and many of those who have come since—initially imagined. 

Despite the common mythos that AI and the data it draws on are objectively and scientifically classifying the world, everywhere there is politics, ideology, prejudices, and all of the subjective stuff of history. When we survey the most widely used training sets, we find that this is the rule rather than the exception.

## Anatomy of a Training Set

Although there can be considerable variation in the purposes and architectures of different training sets, they share some common properties. At their core, training sets for imaging systems consist of a collection of images that have been labeled in various ways and sorted into categories. As such, we can describe their overall architecture as generally consisting of three layers: the overall taxonomy (the aggregate of classes and their hierarchical nesting, if applicable), the individual classes (the singular categories that images are organized into, e.g., “apple,”), and each individually labeled image (i.e., an individual picture that has been labeled an apple). Our contention is that every layer of a given training set’s architecture is infused with politics.

Take the case of a dataset like the “[The Japanese Female Facial Expression (JAFFE) Database](https://www.researchgate.net/figure/Samples-from-the-Japanese-females-facial-expression-image-set_fig4_220013217),” developed by Michael Lyons, Miyuki Kamachi, and Jiro Gyoba in 1998, and widely used in affective computing research and development. The dataset contains photographs of 10 Japanese female models making seven facial expressions that are meant to correlate with seven basic emotional states. The implicit, top-level taxonomy here is something like “facial expressions depicting the emotions of Japanese women.”

There are several implicit assertions in the JAFFE set. First there’s the taxonomy itself: that “emotions” is a valid set of visual concepts. Then there’s a string of additional assumptions: that the concepts within “emotions” can be applied to photographs of people’s faces (specifically Japanese women); that there are six emotions plus a neutral state; that there is a fixed relationship between a person’s facial expression and her true emotional state; and that this relationship between the face and the emotion is consistent, measurable, and uniform across the women in the photographs. Every one of the implicit claims made at each level is, at best, open to question, and some are deeply contested.

## The Canonical Training Set: ImageNet

### Taxonomy

As the fields of information science and science and technology studies have long shown, all taxonomies or classificatory systems are political. In ImageNet (inherited from WordNet), for example, the category “human body” falls under the branch Natural Object > Body > Human Body. Its subcategories include “male body”; “person”; “juvenile body”; “adult body”; and “female body.” The “adult body” category contains the subclasses “adult female body” and “adult male body.” We find an implicit assumption here: only “male” and “female” bodies are “natural.” 

### Categories

There’s a kind of sorcery that goes into the creation of categories. To create a category or to name things is to divide an almost infinitely complex universe into separate phenomena. To impose order onto an undifferentiated mass, to ascribe phenomena to a category—that is, to name a thing—is in turn a means of reifying the existence of that category. 

In the case of ImageNet, noun categories such as “apple” or “apple butter” might seem reasonably uncontroversial, but not all nouns are created equal. To borrow an idea from linguist George Lakoff, the concept of an “apple” is more nouny than the concept of “light”, which in turn is more nouny than a concept such as “health.” Nouns occupy various places on an axis from the concrete to the abstract, and from the descriptive to the judgmental. These gradients have been erased in the logic of ImageNet. Everything is flattened out and pinned to a label, like taxidermy butterflies in a display case. The results can be problematic, illogical, and cruel, especially when it comes to labels applied to people. 

Of course, ImageNet was typically used for object recognition—so the Person category was rarely discussed at technical conferences, nor has it received much public attention. However, this complex architecture of images of real people, tagged with often offensive labels, has been publicly available on the internet for a decade. It provides a powerful and important example of the complexities and dangers of human classification, and the sliding spectrum between supposedly unproblematic labels like “trumpeter” or “tennis player” to concepts like “spastic,” “mulatto,” or “redneck.” Regardless of the supposed neutrality of any particular category, the selection of images skews the meaning in ways that are gendered, racialized, ableist, and ageist. ImageNet is an object lesson, if you will, in what happens when people are categorized like objects. 

Finally, there is the issue of where the thousands of images in ImageNet’s Person class were drawn from. By harvesting images en masse from image search engines like Google, ImageNet’s creators appropriated people’s selfies and vacation photos without their knowledge, and then labeled and repackaged them as the underlying data for much of an entire field.

### Labeled Images

Images are laden with potential meanings, irresolvable questions, and contradictions. In trying to resolve these ambiguities, ImageNet’s labels often compress and simplify images into deadpan banalities. 

At the image layer of the training set, like everywhere else, we find assumptions, politics, and worldviews. According to ImageNet, for example, Sigourney Weaver is a “hermaphrodite,” a young man wearing a straw hat is a “tosser,” and a young woman lying on a beach towel is a “kleptomaniac.” But the worldview of ImageNet isn’t limited to the bizarre or derogatory conjoining of pictures and labels.

Other assumptions about the relationship between pictures and concepts recall physiognomy, the pseudoscientific assumption that something about a person’s essential character can be gleaned by observing features of their bodies and faces. ImageNet takes this to an extreme, assuming that whether someone is a “debtor,” a “snob,” a “swinger,” or a “slav” can be determined by inspecting their photograph. In the weird metaphysics of ImageNet, there are separate image categories for “assistant professor” and “associate professor”—as though if someone were to get a promotion, their biometric signature would reflect the change in rank.

Of course, these sorts of assumptions have their own dark histories and attendant politics.

# UTK: Making Race and Gender from Your Face 

And as we shall see, not only have the underlying assumptions of physiognomy made a comeback with contemporary training sets, but indeed a number of training sets are designed to use algorithms and facial landmarks as latter-day calipers to conduct contemporary versions of craniometry. 

For example, the UTKFace dataset (produced by a group at the University of Tennessee at Knoxville) consists of over 20,000 images of faces with annotations for age, gender, and race. The dataset’s authors state that the dataset can be used for a variety of tasks, like automated face detection, age estimation, and age progression.

The annotations for each image include an estimated age for each person, expressed in years from zero to 116. Gender is a binary choice: either zero for male or one for female. Second, race is categorized from zero to four, and places people in one of five classes: White, Black, Asian, Indian, or “Others.” 

The politics here are as obvious as they are troubling. At the category level, the researchers’ conception of gender is as a simple binary structure, with “male” and “female” the only alternatives. At the level of the image label is the assumption that someone’s gender identity can be ascertained through a photograph.

The classificatory schema for race recalls many of the deeply problematic racial classifications of the twentieth century. Above all, these systems of classifications caused enormous harm to people, and the elusive classifier of a pure “race” signifier was always in dispute. However, seeking to improve matters by producing “more diverse” AI training sets presents its own complications.

# IBM’S Diversity in Faces

IBM’s “Diversity in Faces” dataset was created as a response to critics who had shown that the company’s facial-recognition software often simply did not recognize the faces of people with darker skin. IBM publicly promised to improve their facial-recognition datasets to make them more “representative” and published the “Diversity in Faces” (DiF) dataset as a result. Constructed to be “a computationally practical basis for ensuring fairness and accuracy in face recognition,” the DiF consists of almost a million images of people pulled from the Yahoo! Flickr Creative Commons dataset, assembled specifically to achieve statistical parity among categories of skin tone, facial structure, age, and gender.

The dataset itself continued the practice of collecting hundreds of thousands of images of unsuspecting people who had uploaded pictures to sites like Flickr. But the dataset contains a unique set of categories not previously seen in other face-image datasets. The IBM DiF team asks whether age, gender, and skin color are truly sufficient in generating a dataset that can ensure fairness and accuracy, and concludes that even more classifications are needed. So they move into truly strange territory: including facial symmetry and skull shapes to build a complete picture of the face. The researchers claim that the use of craniofacial features is justified because it captures much more granular information about a person's face than just gender, age, and skin color alone. The paper accompanying the dataset specifically highlights prior work done to show that skin color is itself a weak predictor of race, but this begs the question of why moving to skull shapes is appropriate. 

While the efforts of companies to build more diverse training sets is often put in the language of increasing “fairness” and “mitigating bias, ” clearly there are strong business imperatives to produce tools that will work more effectively across wider markets. However, here too the technical process of categorizing and classifying people is shown to be a political act. For example, how is a “fair” distribution achieved within the dataset?

IBM decided to use a mathematical approach to quantifying “diversity” and “evenness,” so that a consistent measure of evenness exists throughout the dataset for every feature quantified. The dataset also contains subjective annotations for age and gender, which are generated using three independent Amazon Turk workers for each image, similar to the methods used by ImageNet.

Ultimately, beyond these deep methodological concerns, the concept and political history of diversity is being drained of its meaning and left to refer merely to expanded biological phenotyping. Diversity in this context just means a wider range of skull shapes and facial symmetries. For computer vision researchers, this may seem like a “mathematization of fairness” but it simply serves to improve the efficiency of surveillance systems. And even after all these attempts at expanding the ways people are classified, the Diversity in Faces set still relies on a binary classification for gender: people can only be labelled male or female. Achieving parity amongst different categories is not the same as achieving diversity or fairness, and IBM’s data construction and analysis perpetuates a harmful set of classifications within a narrow worldview.

## Epistemics of Training Sets

What are the assumptions undergirding visual AI systems? First, the underlying theoretical paradigm of the training sets assumes that concepts—whether “corn”, “gender,” “emotions,” or “losers”—exist in the first place, and that those concepts are fixed, universal, and have some sort of transcendental grounding and internal consistency. Second, it assumes a fixed and universal correspondences between images and concepts, appearances and essences. What’s more, it assumes uncomplicated, self-evident, and measurable ties between images, referents, and labels. In other words, it assumes that different concepts—whether “corn” or “kleptomaniacs”—have some kind of essence that unites each instance of them, and that that underlying essence expresses itself visually. Moreover, the theory goes, that visual essence is discernible by using statistical methods to look for formal patterns across a collection of labeled images. Finally, this approach assumes that all concrete nouns are created equally, and that many abstract nouns also express themselves concretely and visually.

The training sets of labeled images that are ubiquitous in contemporary computer vision and AI are built on a foundation of unsubstantiated and unstable epistemological and metaphysical assumptions about the nature of images, labels, categorization, and representation. Furthermore, those epistemological and metaphysical assumptions hark back to historical approaches where people were visually assessed and classified as a tool of oppression and race science. 

Datasets aren’t simply raw materials to feed algorithms, but are political interventions. As such, much of the discussion around “bias” in AI systems misses the mark: there is no “neutral,” “natural,” or “apolitical” vantage point that training data can be built upon. There is no easy technical “fix” by shifting demographics, deleting offensive terms, or seeking equal representation by skin tone. The whole endeavor of collecting images, categorizing them, and labeling them is itself a form of politics, filled with questions about who gets to decide what images mean and what kinds of social and political work those representations perform.

## Missing Persons

On one hand, removing these problematic datasets from the internet may seem like a victory. The most obvious privacy and ethical violations are addressed by making them no longer accessible. However, taking them offline doesn’t stop their work in the world: these training sets have been downloaded countless times, and have made their way into many production AI systems and academic papers. By erasing them completely, not only is a significant part of the history of AI lost, but researchers are unable to see how the assumptions, labels, and classificatory approaches have been replicated in new systems, or trace the provenance of skews and biases exhibited in working systems. Facial-recognition and emotion-recognition AI systems are already propagating into hiring, education, and healthcare. They are part of security checks at airports and interview protocols at Fortune 500 companies. Not being able to see the basis on which AI systems are trained removes an important forensic method to understand how they work. This has serious consequences.    

This is the problem of inaccessible or disappearing datasets. If they are, or were, being used in systems that play a role in everyday life, it is important to be able to study and understand the worldview they normalize. Developing frameworks within which future researchers can access these data sets in ways that don’t perpetuate harm is a topic for further work.

## Conclusion: Who decides?

There is much at stake in the architecture and contents of the training sets used in AI. They can promote or discriminate, approve or reject, render visible or invisible, judge or enforce. And so we need to examine them—because they are already used to examine us—and to have a wider public discussion about their consequences, rather than keeping it within academic corridors. As training sets are increasingly part of our urban, legal, logistical, and commercial infrastructures, they have an important but underexamined role: the power to shape the world in their own images.