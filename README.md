<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![My-face-detection](my-face-detection.gif)

This Project detects faces from your live webcam and predicts the age and gender. If you wish to try it online (via web), there soon will be a website that will support this feature. I will keep this project updated by then. So far, you can only try it by cloning this repo to your local computer and by following the instructions below.:smile:

### Built With

* [Python3 (Anaconda based)](https://www.anaconda.com/products/individual)
* [OpenCV](https://jquery.com)
* [Anaconda](https://www.anaconda.com/products/individual)

### Development Environment
* [Ubuntu 20.04 LTS](https://releases.ubuntu.com/20.04/)
* [Anaconda](https://www.anaconda.com/products/individual)
* [Visual Studio Code](https://code.visualstudio.com/)


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you can go about setting up this project locally. 
To get a local copy up and running follow these simple example steps.

### Prerequisites


* Webcam 
    * You need a webcam for the project to detect faces.
* Anaconda (If you don't have it installed, please follow through the installation steps below.)
    * Anaconda is a free and open-source Python distribution and collection of hundreds of packages related to data science, scientific programming, development and more.

### Installation

1. Download [Anaconda](https://www.anaconda.com/products/individual)
2. Clone the repo
   ```sh
   git clone https://github.com/PricelessCode/Age-Gender-Detector.git
   ```
3. Create virtual environment for this project (**Optional**)
    * Create virtual env 
        ```sh
        conda create age-gender-detector
        ```
    * Activate the virtual env

        ```sh
        source activate age-gender-detector
        ```
    
    Then you will see the activated virtual environment with parenthesis in your terminal.

3. Install required packages via Anaconda
   ```sh
   conda install --file requirements.txt
   ```

   I have exported all the required packages for this project in `requirements.txt`. Running the command above will automatically install the required packages by itself.

4. Go to the project folder and run the project!
    * Go to the project folder
        ```sh
        cd Age-Gender-Detector
        ```

    * Run the detect_age_gender_video.py
        ```sh
        python3 detect_age_gender_video.py
        ```



<!-- USAGE EXAMPLES -->
## Usage / Review

* It detects faces in real time and predict age and gender.
* I am 22 years old by the time I recorded this. So the prediction is quite reliable.

    ![My-face-detection](my-face-detection.gif)


* It can also detect multiple faces and faces from images and videos. (I got the image in the phone from Google)

    ![Other-face-detections](other-face-detection.gif)


## How it works under the hood
1. Face Detection
    * The face is detected based on OpenCV's Haar feature-based cascade classifiers.

2. Age Prediction
    * The age is predicted based on [Levi and Tal Hassner's Age Classification model](https://talhassner.github.io/home/publication/2015_CVPR)

    * Ideally, Age Prediction should be approached as a Regression problem since we are expecting a real number as the output. However, estimating age accurately using regression is challenging. Even humans cannot accurately predict the age based on looking at a person. However, we have an idea of whether they are in their 20s or in their 30s. Because of this reason, it is wise to frame this problem as a classification problem where we try to estimate the age group the person is in. For example, age in the range of 0-2 is a single class, 4-6 is another class and so on.

    * The Adience dataset has 8 classes divided into the following age groups [(0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100)]. Thus, the age prediction network has 8 nodes in the final softmax layer indicating the mentioned age ranges.

    * It should be kept in mind that Age prediction from a single image is not a very easy problem to solve as the perceived age depends on a lot of factors and people of the same age may look pretty different in various parts of the world. Also, people try very hard to hide their real age!

3. Gender Prediction
    * The Gender is predicted based on [Levi and Tal Hassner's Gender Classification model](https://talhassner.github.io/home/publication/2015_CVPR)

    * They have framed Gender Prediction as a classification problem. The output layer in the gender prediction network is of type softmax with 2 nodes indicating the two classes “Male” and “Female”.




<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**. Also, Feel free to raise issues on the [issue page](https://github.com/PricelessCode/Age-Gender-Detector/issues)

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Seungho Lee - poream3387@gmail.com



<!-- ACKNOWLEDGEMENTS -->
## Resource / Acknowledgements
* [Gil Levi and Tal Hassner's Pre-trained Models](https://talhassner.github.io/home/publication/2015_CVPR)
* [Age detection Idea](https://www.pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/)
* [Article about the models](https://www.pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/)
