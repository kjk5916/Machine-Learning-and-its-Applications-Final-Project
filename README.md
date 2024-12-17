# Machine-Learning-and-its-Applications-Final-Project
Final Project for the 2024-2 Machine Learning and its Applications Course

안녕하세요, 전기전자공학부/수학과 복수전공중인 학부생 2018142087 김준규 입니다.

이번 프로젝트에서는, 제가 기존에 했던 Continual Learning 연구에 이어 Unsupervised Continual Learning에 대한 연구를 진행하였습니다. 구체적으로는, 아래 paper에서 제시한 method에서 projector network를 수정하여 unsupervised continual learning의 performance 향상을 시킬 수 있는가에 대해 알아보았습니다. 

[Continually Learning Self-Supervised Representations with Projected Functional Regularization
Alex Gomez-Villa, Bartlomiej Twardowski, Lu Yu, Andrew D. Bagdanov, Joost van de Weijer
CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/html/Gomez-Villa_Continually_Learning_Self-Supervised_Representations_With_Projected_Functional_Regularization_CVPRW_2022_paper.html)

![pfr](https://github.com/user-attachments/assets/641457e3-299c-4fc2-aa18-bee8967ecf20)

Unsupervised Continual Learning에서는 forgetting을 방지하기 위해 주로 previous time step 모델에서 현재 time step 모델로 knowledge distillation을 합니다. 그러나, 직접적으로 distillation을 구현할 경우, 모델이 현재 task를 잘 학습하지 못하는 현상이 발생하게 됩니다. 이에 대한 해결책으로 위 논문에서는 distillation 과정에서 2-layer MLP projector network을 사용하는 방식을 제안합니다. 하지만, 2-layer MLP projector network은 현재 나오는 deep learning 모델들에 비해 expressivity가 현저히 작다는 문제점이 있습니다. 따라서 제가 제안한 방식은, 2-layer MLP projector network을 Transformer 모델로 바꿔보는 것 입니다. 이를 위해 기존 논문에서 사용했던 code에 projector network을 transformer로 수정하여 실험을 진행하였습니다.

데이터셋은 CiFAR100 dataset을 사용하였으며, 5개의 class씩 하나의 task를 구성하여 총 20개의 task로 unsupervised continual learning 실험을 진행하였습니다.

이에 대한 결과는 다음과 같습니다.

