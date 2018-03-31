#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Windows.h>
#pragma warning(disable:4996)

#define Train 1
#define Test 0
#define c 0.01
//Layer 수
#define NLayer 3

//각 층의 뉴런 수
#define m0 3
#define m1 2
#define m2 3
int M[NLayer] = { m0, m1, m2 }; // 전층의 뉴런수를 파악하기 위한 변수

								// 입력 신호의 수
#define N 65 

#define N_tr_examples 600
int d_tr[N_tr_examples][m2];
int TrainData[N_tr_examples][N];

#define N_te_examples 90
int d_te[N_te_examples][m2];
int TestData[N_te_examples][N];

double s[NLayer][N];  // weighted sum
double f[NLayer][N];  // 각 뉴런의 출력
double delta[NLayer][N]; // 각 뉴런이 가지고 있는 delta
double W[NLayer][N][N]; // N번째 층에, M번째 뉴런에서 다음 뉴런으로 연결되는 선의 가중치벡터

void setDesiredValue(int trNum, int whatNum, int whatData);
void initial_S_F_Delta();


void forwardTrain(int trNum);
void forwardTest(int trNum);
void backward(int trNum);
void renewWeight(int trNum);

void initialRandomWeight();
void loadTrainData();
void loadTestData();


void main()
{
	double sumSqError = 0;
	double avgSqError = 0;
	int correctCount = 0;
	double accuracy = 0;
	int epochCount = 0;
	int sysOut[m2];

	initialRandomWeight();
	initial_S_F_Delta();

	loadTrainData();
	loadTestData();
	
	do {
		sumSqError = 0;
		avgSqError = 0;

		for (int i = 0; i < N_tr_examples; i++)
		{
			forwardTrain(i);
			backward(i);
			renewWeight(i);
		}// 전체 수행하면 한개의 에폭(600개의 훈련예제를 수행한 것)을 수행한 것이됨.

		epochCount++;

		for (int t = 0; t < N_tr_examples; t++)
		{
			forwardTrain(t);

			for (int i = 0; i < M[NLayer - 1]; i++)
			{
				sumSqError += pow((d_tr[t][i] - f[NLayer - 1][i]), 2);
			}
		}

		avgSqError = sumSqError / (N_tr_examples * M[NLayer - 1]);
		printf("%d epoch avgSqError: %f\n\n", epochCount, avgSqError);
	} while (avgSqError > 0.01);
	

	forwardTrain(0);
	backward(0);
	renewWeight(0);


	//테스트 과정
	for (int t = 0; t < N_te_examples; t++)
	{
		initial_S_F_Delta();

		forwardTest(t);

		//printf("f[NLayer - 1][0] = %f\n", f[NLayer - 1][0]);
		//printf("f[NLayer - 1][1] = %f\n", f[NLayer - 1][1]);
		//printf("f[NLayer - 1][2] = %f\n", f[NLayer - 1][2]);

		//디폴트로 0이라고 생각하고 시작.
		sysOut[0] = 1;
		sysOut[1] = 0;
		sysOut[2] = 0;

		if (f[NLayer - 1][0] < f[NLayer - 1][1])
		{
			sysOut[0] = 0;
			sysOut[1] = 1;
			sysOut[2] = 0;

			if (f[NLayer - 1][1] < f[NLayer - 1][2])
			{
				sysOut[0] = 0;
				sysOut[1] = 0;
				sysOut[2] = 1;
			}
		}

		if (d_te[t][0] == sysOut[0] && d_te[t][1] == sysOut[1] && d_te[t][2] == sysOut[2])
		{
			correctCount++;
		}
		//printf("sysOut[0] = %d\n", sysOut[0]);
		//printf("sysOut[1] = %d\n", sysOut[1]);
		//printf("sysOut[2] = %d\n", sysOut[2]);

	}

	accuracy = (double)correctCount / N_te_examples;

	printf("0층/1층/2층 뉴론의 수 : %d / %d / %d \n", m0, m1, m2);
	printf("학습률 : %f\n", c);
	printf("사용된 epoch 수 : %d\n", epochCount);
	printf("test accuracy : %f\n\n", accuracy);
	//Sleep(1000);

	getchar();
}

void initialRandomWeight()
{
	int preLayer;
	double randomValue;
	double signedRandomValue;

	srand(time(NULL));

	for (int i = 0; i < NLayer; i++) // i = 0, 1, 2
	{
		// i층 전층의 뉴론의 수를 가져옴.
		if (i == 0)
			preLayer = N; // 0~64
		else
			preLayer = M[i - 1] + 1;

		for (int j = 0; j < M[i]; j++)
		{
			for (int k = 0; k < preLayer; k++)
			{
				randomValue = double(rand());
				signedRandomValue = double(rand());
				if (signedRandomValue < RAND_MAX / 2)
					W[i][j][k] = -1 * (randomValue / double(RAND_MAX));
				else
					W[i][j][k] = randomValue / double(RAND_MAX);
				//printf("W[%d][%d][%d] : %f\n", i, j, k, W[i][j][k]);
			}
		}
	}
	//printf("1. 웨이트가 랜덤으로 초기화 되었습니다.\n");
}//initialRandomWeight()
void initial_S_F_Delta()
{
	for (int i = 0; i < NLayer; i++)
	{
		for (int j = 0; j < N; j++)
		{
			s[i][j] = 0;
			f[i][j] = 0;
			delta[i][j] = 0;
		}
	}
}

void loadTrainData()
{
	int trNum = 0;	  // 훈련예제 번호
	int whatNum;	  // 각 트레이닝이 의미하는 숫자
	char separator;   // 쓰레기 문자를 담을 변수

	FILE *fp;
	fp = fopen("traindata.txt", "r");

	if (fp == NULL)
	{
		printf("파일의 경로를 다시 확인하시기 바랍니다.\n");
		Sleep(1000);
		exit(1);
	}
	else
	{
		while (!feof(fp))
		{
			fscanf(fp, "%d %c", &whatNum, &separator); // 한 줄의 첫 글자를 읽어들이고, 그 다음 '$'(버리는 문자)도 읽어들인다. 

			setDesiredValue(trNum, whatNum, Train);		   // 읽어들인 첫 글자가 무엇인지에 따라 각 예제에 해당하는 desired 값을 넣어준다.

			for (int i = 0; i < N; i++)
				fscanf(fp, "%d", &TrainData[trNum][i]);

			trNum++;
		}
	}
}//loadTrainData()
void loadTestData()
{
	int trNum = 0; // 훈련예제번호
	int whatNum;	  // 각 트레이닝이 의미하는 숫자
	char separator;   // 쓰레기 문자를 담을 변수

	FILE *fp;
	fp = fopen("testdata.txt", "r");

	if (fp == NULL)
	{
		printf("파일의 경로를 다시 확인하시기 바랍니다.\n");
		Sleep(1000);
		exit(1);
	}
	else
	{
		while (!feof(fp))
		{
			fscanf(fp, "%d %c", &whatNum, &separator); // 한 줄의 첫 글자를 읽어들이고, 그 다음 '$'(버리는 문자)도 읽어들인다. 

			setDesiredValue(trNum, whatNum, Test);		   // 읽어들인 첫 글자가 무엇인지에 따라 각 예제에 해당하는 desired 값을 넣어준다.

			for (int i = 0; i < N; i++)
				fscanf(fp, "%d", &TestData[trNum][i]);

			trNum++;
		}
	}
}
void setDesiredValue(int trNum, int whatNum, int whatData)
{
	if (whatData == Train)
	{
		if (whatNum == 0)
		{
			d_tr[trNum][0] = 1;
			d_tr[trNum][1] = 0;
			d_tr[trNum][2] = 0;
		}

		else if (whatNum == 1)
		{
			d_tr[trNum][0] = 0;
			d_tr[trNum][1] = 1;
			d_tr[trNum][2] = 0;
		}

		else if (whatNum == 2)
		{
			d_tr[trNum][0] = 0;
			d_tr[trNum][1] = 0;
			d_tr[trNum][2] = 1;
		}
	}
	else if (whatData == Test)
	{
		if (whatNum == 0)
		{
			d_te[trNum][0] = 1;
			d_te[trNum][1] = 0;
			d_te[trNum][2] = 0;
		}

		else if (whatNum == 1)
		{
			d_te[trNum][0] = 0;
			d_te[trNum][1] = 1;
			d_te[trNum][2] = 0;
		}

		else if (whatNum == 2)
		{
			d_te[trNum][0] = 0;
			d_te[trNum][1] = 0;
			d_te[trNum][2] = 1;
		}
	}

}//setDesiredValue()

void forwardTrain(int trNum) {
	initial_S_F_Delta();

	// 0번째 층 s,f값 구하기
	for (int i = 0; i<M[0]; i++)
	{
		for (int j = 0; j<N; j++)
		{
			s[0][i] += (double)TrainData[trNum][j] * W[0][i][j];
		}
		f[0][i] = 1.0 / (1 + exp(-s[0][i]));
	}

	// 나머지 층 s,f 값 구하기
	for (int L = 1; L<3; L++) {
		for (int i = 0; i<M[L]; i++) {
			for (int j = 0; j<M[L - 1] + 1; j++) {
				if (j == M[L - 1])
					f[L - 1][j] = 1;
				s[L][i] += f[L - 1][j] * W[L][i][j];
			}
			f[L][i] = 1 / (1 + exp(-s[L][i]));
		}
	}
}
void forwardTest(int trNum) {
	initial_S_F_Delta();

	// 0번째 층 s,f값 구하기
	for (int i = 0; i<M[0]; i++)
	{
		for (int j = 0; j<N; j++)
		{
			s[0][i] += (double)TestData[trNum][j] * W[0][i][j];
		}
		f[0][i] = 1.0 / (1 + exp(-s[0][i]));
	}

	// 나머지 층 s,f 값 구하기
	for (int L = 1; L<3; L++) {
		for (int i = 0; i<M[L]; i++) {
			for (int j = 0; j<M[L - 1] + 1; j++) {
				if (j == M[L - 1])
					f[L - 1][j] = 1;
				s[L][i] += f[L - 1][j] * W[L][i][j];
			}
			f[L][i] = 1 / (1 + exp(-s[L][i]));
		}
	}
}
void backward(int trNum)
{
	int finalLayer = NLayer - 1;
	double tempSum = 0;

	// 최종층에 대한 delta 값 구하기
	for (int i = 0; i<M[finalLayer]; i++) {
		delta[finalLayer][i] = (d_tr[trNum][i] - f[finalLayer][i])*f[finalLayer][i] * (1 - f[finalLayer][i]);
	}

	// 나머지 층에 대한 delta 값 구하기
	for (int L = NLayer - 2; L >= 0; L--)
	{
		for (int i = 0; i<M[L]; i++)
		{
			for (int j = 0; j<M[L + 1]; j++) {
				tempSum += delta[L + 1][j] * W[L + 1][j][i];
			}
			delta[L][i] = f[L][i] * (1 - f[L][i]) * tempSum;
		}
	}
}
void renewWeight(int trNum) {
	// 맨 첫 W갱신
	for (int i = 0; i < M[0]; i++)
	{
		for (int j = 0; j < N; j++) {
			W[0][i][j] += c * delta[0][i] * TrainData[trNum][j];
		}
	}

	// 나머지 W갱신
	for (int layer = 1; layer <= NLayer - 1; layer++)
	{
		for (int i = 0; i < M[layer]; i++)
		{
			for (int j = 0; j < M[layer - 1] + 1; j++)
			{
				W[layer][i][j] += c*delta[layer][i] * f[layer - 1][j];
			}//더미입력까지 여기서 처리.
		}
	}
}

