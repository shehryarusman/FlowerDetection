# Flower Classification using Transfer Learning

This project applies transfer-learning to build an Flower classification model.
Specifically we are using the MobileNetV2 model as the base and retrain it on the
Oxford Flower 102 dataset before applying further fine-tuning. Finally we
deployed the model as an app for ios devices

## Post
For further reading/ understanding of this project, find the related
blog-post on my [website](https://paul-mora.com)

## Installation

All relevant packages are stated within the requirements file as well as pipfiles,
which were used for the project.

```Usage in bash
python3 main.py --config ./src/config.json
```

## Preview

<img src="https://github.com/paulmora-statworx/flower_detection/blob/main/reports/gif/testing_gif.gif" width="250"/>

## Contributing
Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)