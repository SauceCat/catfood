# Hierarchical classification

To build this tree we examine the visual nouns in ImageNet and look at their paths through the WordNet graph to the root node, in this case, “physical object”.

- Many synsets only have one path through the graph. We first add all of these paths to our tree.
- Then we iteratively examine the rest concepts and add the paths that grow the tree as little as possible. So if a concept has multiple paths to the root, we choose the shortest one.

The final result is **WordTree**, a hierarchical model of visual concepts. To perform classification with WordTree we predict conditional probabilities at each node. For example, at the “terrier” node we predict:

![](/home/saucecat/Desktop/develop/catfood/object_detection/yolo/images/yolov2_wordtree_node.png)

If we want to compute the absolute probability for a particular node we simply follow the path through the tree to the root node and multiply by conditional probabilities. For classification purposes, we assume that the image contains an object: `Pr(physical object) = 1`.

![](/home/saucecat/Desktop/develop/catfood/object_detection/yolo/images/yolov2_wordtree_node_2.png)

To validate this approach, we train the Darknet-19 model on WordTree built from the 1000-class ImageNet and named it **WordTree1k**. To build **WordTree1k**, we add all the intermediate nodes and expands the label space from `1000` to `1369`.

During training, we propagate ground truth labels up the tree. Therefore, if an image is labeled as a “Norfolk terrier”, it also gets labeled as a “dog” and a “mammal”, etc. To compute the conditional probabilities, our model predicts a vector of `1369` values and we compute the softmax over all synsets that are hyponyms of the same concept.

![](/home/saucecat/Desktop/develop/catfood/object_detection/yolo/images/yolov2_wordtree1k.png)

**Figure 5:** Prediction on ImageNet vs WordTree. Most ImageNet models use one large softmax to predict a probability distribution. Using WordTree we perform multiple softmax operations over co-hyponyms.

Using the same training parameters as before, our hierarchical Darknet-19 achieves `71.9%` top-1 accuracy and `90.4%` top-5 accuracy, comparing with the original performance of `72.9%` top-1 accuracy and `91.2%` top-5 accuracy. Despite adding `369` additional concepts and having our network predict a tree structure, our accuracy only drops marginally.

Besides, performing classification in this manner enables performance to degrade gracefully on new or unknown object categories. For example, if the network sees a picture of a dog but is uncertain about the detailed dog breed, it will still predict “dog” with high confidence but have lower confidences spread out among the hyponyms.

This formulation also works for detection. Now, instead of assuming every image has an object, we use YOLOv2’s objectness predictor to give us the value of `Pr(physical object)`. The detector predicts a bounding box and the tree of probabilities. We traverse the tree down, taking the highest confidence path at every split until we reach some threshold and we predict that object class.

## implementation

**Reference:** https://github.com/jkvt2/hierloss/blob/master/hierarchical_loss.py

### Sample dummy WordTree

```python
word_tree = {
    'animal': {
        'cat': {
            'big-cat': {
                'lion': '', 
                'tiger': ''
            },
            'small-cat': ''
        },
        'dog': {
            'collie': '', 
            'dalmatian': '', 
            'terrier': ''
        },
        'mouse': ''
    },
    'elements': {
        'acid': {
            'h2so4': '', 
            'hcl': ''
        },
        'base': {
            'strong': {
                'koh': '', 
                'naoh': ''
            },
            'weak': {
                'ch3nh2': '', 
                'nh3': '', 
                'nh4oh': ''
            }
        }
    }
}
```

### WordTree Loss

```python
def wordtree_loss(logits, labels, word_tree, epsilon = 1e-5):
    '''
    Builds the wordtree style loss function as described in YOLO9000
    (https://arxiv.org/abs/1612.08242)
    Args:
        logits (tf.Tensor): Classification logits.
        labels (tf.Tensor): The one hot tensor of the ground truth labels.
        word_tree (dict): Dictionary of dictionaries showing the relationship between the classes.
        epsilon (float, optional): Epsilon term added to make the softmax cross entropy stable. Defaults to 1e-5.
    Returns:
        loss: Tensor of shape (batch_size, ), giving the loss for each example.
        raw_probs: The probability for each class (given its parents).
    '''
    
# shapes:
logits.shape = (batch_size, num_classes)
labels.shape = (batch_size, num_classes)
```

`n`:

```python
class_list, n = _get_dict_item(word_tree)

# values
class_list = [
    'animal',
    'elements',
    'animal/cat',
    'animal/dog',
    'animal/mouse',
    'animal/cat/big-cat',
    'animal/cat/small-cat',
    'animal/cat/big-cat/lion',
    'animal/cat/big-cat/tiger',
    'animal/dog/collie',
    'animal/dog/dalmatian',
    'animal/dog/terrier',
    'elements/acid',
    'elements/base',
    'elements/acid/h2so4',
    'elements/acid/hcl',
    'elements/base/strong',
    'elements/base/weak',
    'elements/base/strong/koh',
    'elements/base/strong/naoh',
    'elements/base/weak/ch3nh2',
    'elements/base/weak/nh3',
    'elements/base/weak/nh4oh'
]
n = [[[[0, 0], 0], [0, 0, 0], 0], [[0, 0], [[0, 0], [0, 0, 0]]]]
```

`n_flat`:

```python
n_flat = [len(n)] + list(_flatten(n))

# values
n_flat = [
    2, # 0   root: [animal, elements]
    3, # 1   animal: [cat, dog, mouse]
    2, # 2   cat: [big-cat, small-cat]
    2, # 3   big-cat: [lion, tiger]
    0, # 4   small-cat: ''
    0, # 5   lion: ''
    0, # 6   tiger: ''
    3, # 7   dog: [collie, dalmatian, terrier]
    0, # 8   collie: ''
    0, # 9   dalmatian: ''
    0, # 10  terrier: ''
    0, # 11  mouse: ''
    2, # 12  elements: [acid, base]
    2, # 13  acid: [h2so4, hcl]
    0, # 14  h2so4: ''
    0, # 15  hcl: ''
    2, # 16  base: [strong, weak]
    2, # 17  strong: [koh, naoh]
    0, # 18  koh: ''
    0, # 19  naoh: ''
    3, # 20  weak: [ch3nh2, nh3, nh4oh]
    0, # 21  ch3nh2: ''
    0, # 22  nh3: ''
    0  # 23  nh4oh: ''
]
```

`parents`:

```python
parents, children, childless = _get_idxs(n_flat)

# values
# length: 23
parents = [
    [0],
    [1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 2, 5],
    [0, 2, 6],
    [0, 2, 5, 7],
    [0, 2, 5, 8],
    [0, 3, 9],
    [0, 3, 10],
    [0, 3, 11],
    [1, 12],
    [1, 13],
    [1, 12, 14],
    [1, 12, 15],
    [1, 13, 16],
    [1, 13, 17],
    [1, 13, 16, 18],
    [1, 13, 16, 19],
    [1, 13, 17, 20],
    [1, 13, 17, 21],
    [1, 13, 17, 22]
]
# length: 23
children = [
    [2, 3, 4],
    [12, 13],
    [5, 6],
    [9, 10, 11],
    [],
    [7, 8],
    [],
    [],
    [],
    [],
    [],
    [],
    [14, 15],
    [16, 17],
    [],
    [],
    [18, 19],
    [20, 21, 22],
    [],
    [],
    [],
    [],
    []
]
# length: 14
childless = [7, 8, 6, 9, 10, 11, 4, 14, 15, 18, 19, 20, 21, 22]
```

`subsoftmax_idx`:

```python
subsoftmax_idx = np.cumsum([0] + n_flat, dtype = np.int32)

# values
# length: 25
subsoftmax_idx = array([ 0,  2,  5,  7,  9,  9,  9,  9, 12, 12, 12, 12, 12, 14, 16, 16, 16, 18, 20, 20, 20, 23, 23, 23, 23], dtype=int32)

raw_probs = tf.concat([
    tf.nn.softmax(
        logits[:, subsoftmax_idx[i]: subsoftmax_idx[i + 1]]
    ) 
    for i in range(len(n_flat))
], 1)

# values
subsoftmax_indices = [
    (0, 2),   # root: [animal, elements]
    (2, 5),   # animal: [cat, dog, mouse]
    (5, 7),   # cat: [big-cat, small-cat]
    (7, 9),   # big-cat: [lion, tiger]
    (9, 9),   # small-cat: ''
    (9, 9),   # lion: ''
    (9, 9),	  # tiger: ''
    (9, 12),  # dog: [collie, dalmatian, terrier]
    (12, 12), # collie: ''
    (12, 12), # dalmatian: ''
    (12, 12), # terrier: ''
    (12, 12), # mouse: ''
    (12, 14), # elements: [acid, base]
    (14, 16), # acid: [h2so4, hcl]
    (16, 16), # h2so4: ''
    (16, 16), # hcl: ''
    (16, 18), # base: [strong, weak]
    (18, 20), # strong: [koh, naoh]
    (20, 20), # koh: ''
    (20, 20), # naoh: ''
    (20, 23), # weak: [ch3nh2, nh3, nh4oh]
    (23, 23), # ch3nh2: '' 
    (23, 23), # nh3: ''
    (23, 23)  # nh4oh: ''
]
```

`loss`:

```python
probs = tf.concat([
    tf.reduce_prod(
        tf.gather(raw_probs, p, axis=1),
        axis=1, keepdims=True
    ) for p in parents
], 1)

loss = tf.reduce_sum(-tf.log(probs + epsilon) * labels, 1)
```

