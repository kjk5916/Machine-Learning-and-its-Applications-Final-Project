# Machine-Learning-and-its-Applications-Final-Project
Final Project for the 2024-2 Machine Learning and its Applications Course

안녕하세요, 전기전자공학부/수학과 복수전공중인 학부생 2018142087 김준규 입니다.

이번 프로젝트에서는, 제가 기존에 했던 Continual Learning 연구에 이어 Unsupervised Continual Learning에 대한 연구를 진행하였습니다. 구체적으로는, 아래 paper에서 제시한 method에서 projector network를 수정하여 unsupervised continual learning의 performance 향상을 시킬 수 있는가에 대해 알아보았습니다. 

[Continually Learning Self-Supervised Representations with Projected Functional Regularization
Alex Gomez-Villa, Bartlomiej Twardowski, Lu Yu, Andrew D. Bagdanov, Joost van de Weijer
CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/html/Gomez-Villa_Continually_Learning_Self-Supervised_Representations_With_Projected_Functional_Regularization_CVPRW_2022_paper.html)

![pfr](https://github.com/user-attachments/assets/641457e3-299c-4fc2-aa18-bee8967ecf20)

Unsupervised Continual Learning에서는 forgetting을 방지하기 위해 주로 previous time step 모델에서 현재 time step 모델로 knowledge distillation을 합니다. 그러나, 직접적으로 distillation을 구현할 경우, 모델이 현재 task를 잘 학습하지 못하는 현상이 발생하게 됩니다. 이에 대한 해결책으로 위 논문에서는 distillation 과정에서 2-layer MLP projector network을 사용하는 방식을 제안합니다. 하지만, 2-layer MLP projector network은 현재 나오는 deep learning 모델들에 비해 expressivity가 현저히 작다는 문제점이 있습니다. 따라서 제가 제안한 방식은, 2-layer MLP projector network을 Transformer 모델로 바꿔보는 것 입니다. 이를 위해 기존 논문에서 사용했던 code에 projector network을 transformer로 수정하여 실험을 진행하였습니다.

#실험 Setting: 

데이터셋은 CiFAR100 dataset을 사용하였으며, 5개의 class씩 하나의 task를 구성하여 총 20개의 task로 unsupervised continual learning 실험을 진행하였습니다.
Proposed method에 대한 비교를 위해서 다음과 같은 모델들을 사용하였습니다.

Finetuning: Distillation projector 없이 그냥 finetuning을 통해 unsupervised continual learning을 한 모델 입니다.

PFR: 기존 논문에서 사용했던 2-layer MLP projector 입니다. layer당 dimension은 512-256-512 입니다.

Wide-PFR: 2-layer projector에서 중간 layer의 dimension을 8배 높인 projector 입니다. 따라서 dimension은 512-2048-512 입니다.

Deep-PFR: 기존 pfr에서 layer depth를 5로 높인 projector 입니다. 따라서 dimension은 512-256-256-256-256-256-512 로 구성되어 있습니다.

TPFR: MLP layer 대신 Vision Transformer Block을 이용해서 projector을 구성했습니다. 여기서 한가지 주목할 점은, transformer block은 mlp와 다르게 input 단에서 sequence length라는 새로운 dimension이 필요합니다. 이를 위해 기존에 사용되었던 ResNet18 encoder output에서 Global Average Pooling layer을 통과하기 전, size 25 by 25 by 512 input을 사용하였습니다. 즉, 25 by 25 feature map들을 각각 하나의 image patch로 보고, 512를 sequence dimension으로 본 것 입니다.

#Result:

![image](https://github.com/user-attachments/assets/b743672f-8c61-48cc-842a-96a6c5b2fa1b)

위 결과를 통해 볼 수 있는 것은, projector network을 transformer network로 변형시켰을 때 실제로 1%정도의 performance 이득이 있었다는 것 입니다. 특히, 초반에는 TPFR이 기존의 다른 MLP류의 projector에 비해 낮은 performance를 보이다가, 이후에는 더 좋은 성능을 내는 것을 확인할 수 있습니다. 제가 생각했을 때 그 이유로는 transformer network이 mlp에 비해 학습 속도가 늦어져서 그렇지 않을까 싶습니다. 이를 통해 알 수 있는 것은, projector network을 무한정으로 늘리는 것은 오히려 continual learning에 방해가 될 수 도 있다는 것 입니다.

감사합니다.
