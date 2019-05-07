#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  

using namespace cv;

#define PI 3.14159265359

typedef struct {
	int r, g, b;
}int_rgb;


int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

float** FloatAlloc2(int height, int width)
{
	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i < height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0까지 포함된 갯수임

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

void main_0305()
{
	int height, width;
	int** image = ReadImage((char*)"pG.jpg", &height, &width);

	//printf("height = %d, width = %d", height, width);
	for (int y = 0; y < 300; y++)
		for (int x = 0; x < 300; x++)
		{
			image[y][x] = 255;
		}

	ImageShow((char*)"test", image, height, width);

}
void main_0312()//0312
{
	int height, width;
	int** image = ReadImage((char*)"pG.jpg", &height, &width);

	//printf("height = %d, width = %d", height, width);
	int thick = 20;
	for (int y = 0; y < 300; y++)
		for (int x = 0; x < 300; x++)
		{
			for (int z = (-1)*thick; z <= thick; z++)
			{
				y = (int)(2 * x + 100 + 0.5);
				//image[y+z][x+z] = 255; 예외처리를 해 주어야 한다(width-1, height-1를 넘는 경우)
				if (y < 0 || y >= height - 1) continue;
				else image[y + z][x + z] = 255;
			}
		}

	ImageShow((char*)"test", image, height, width);

}

void DrawLine(int**image, int height, int width, double a, double b,
	double Thickness)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			double d = fabs(a*x - y + b) / sqrt(a*a + 1.0);

			if (d < Thickness) {
				image[y][x] = 0;
			}
		}
	}


}
void DrawCircle(int**image, int height, int width, int a, int b, double r) //(a,b)
{

	for (int x = -r; x < r; x++)
	{
		int y = (int)(sqrt(r*r - x * x));
		image[b + y][a + x] = 255;
		image[b - y][a + x] = 255;
	}

	for (int y = -r; y < r; y++)
	{
		int x = (int)(sqrt(r*r - y * y));
		image[b + y][a + x] = 255;
		image[b + y][a - x] = 255;
	}
}

void DrawCircleFull(int**image, int height, int width, int a, int b, double r) //(a,b)
{
	for (int x = -r; x < r; x++)
	{
		int y = (int)(sqrt(r*r - x * x));
		for (int i = -y; i < y; i++)
		{
			image[b + i][a + x] = 255;
		}
	}
}


void main_0312_2()//점과 직선 사이의 공식 이용
{
	int height, width;
	int** image = ReadImage((char*)"pG.jpg", &height, &width);

	//y= ax+b -> ax-y+b = 0 -> d = |ax0 - y0 + b| / sqrt(a*a + 1)
	//함수화. 입력 : a, b, Thickness, image, width, height
	double a = 1.0;
	double b = 50.0;
	double Thickness = 3.0;
	DrawLine(image, height, width, a, b, Thickness);
	DrawCircle(image, height, width, width / 2, height / 2, 100);
	DrawCircleFull(image, height, width, width / 2, height / 2, 50);
	//	for(int i=0.1; i<10; i+=0.1)DrawLine(image, height, width, i, b, Thickness);


	ImageShow((char*)"test", image, height, width);
}

void main_0313()
{
	//03/13 affine transform

	int height, width;
	int** image = ReadImage((char*)"pG.jpg", &height, &width);
	int** image_out = IntAlloc2(height, width);
	float a, b, c, d, t1, t2;
	a = -1; b = 0; c = 0; d = -1;
	//t1 = 0; t2 = 0;
	t1 = width, t2 = height;


	for (int i = 1; i < height; i++)
		for (int j = 1; j < width; j++)
		{
			if (a*j + b * i + t1 >= width || c * j + d * i + t2 >= height || a * j + b * i + t1 < 0 || c * j + d * i + t2 < 0) continue;
			else image_out[(int)(c*j + d * i + t2)][(int)(a*j + b * i + t1)] = image[i][j];
		}


	ImageShow((char*)"test", image, height, width);
	ImageShow((char*)"test", image_out, height, width);
}


#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)
/*
int BilinearInterpolation(int** image, int width, int height, double x, double y)
{
int x_int = (int)x;
int y_int = (int)y;

int A = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
int B = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];
int C = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
int D = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];

double dx = x - x_int;
double dy = y - y_int;

double value
= (1.0 - dx)*(1.0 - dy)*A + dx*(1.0 - dy)*B
+ (1.0 - dx)*dy*C + dx*dy*D;

return((int)(value + 0.5));
}
*/


int BilinearInterpolation(int**image, int width, int height, double xx, double yy)
{
	int A_x = (int)xx; int A_y = (int)yy;
	int B_x = A_x + 1; int B_y = A_y;
	int C_x = A_x; int C_y = A_y + 1;
	int D_x = A_x + 1; int D_y = A_y + 1;


#if 0
	A_x = imin(imax(A_x, 0), width - 1);	B_x = imin(imax(B_x, 0), width - 1);	C_x = imin(imax(C_x, 0), width - 1);	D_x = imin(imax(D_x, 0), width - 1);
	A_y = imin(imax(A_y, 0), height - 1);	B_y = imin(imax(B_y, 0), height - 1);	C_y = imin(imax(C_y, 0), height - 1);	D_y = imin(imax(D_y, 0), height - 1);

#else
	if (A_x<0 || A_x > width - 1 || B_x<0 || B_x > width - 1 || C_x<0 || C_x > width - 1 || D_x<0 || D_x > width - 1
		|| A_y<0 || A_y > height - 1 || B_y<0 || B_y > height - 1 || C_y<0 || C_y > height - 1 || D_y<0 || D_y > height - 1)
		return (0);

#endif

	double dx = xx - A_x;
	double dy = yy - A_y;

	int X = image[A_y][A_x] * (1 - dx) * (1 - dy) + image[B_y][B_x] * dx * (1 - dy) + image[C_y][C_x] * (1 - dx) * dy + image[D_y][D_x] * dx * dy;

	return X;
}


/* 03/19
쌍선형 보간 Bilinear Interpolation
x = 1/(ad-bc) *{d(x'-t1) - b*(y'-t2)}
y = 1/(ad-bc) *{-c(y'-t1) + a*(y'-t2)}
*/
void main_0319()
{
	int height, width;
	int** image = ReadImage((char*)"pG.jpg", &height, &width);
	int** image_out = IntAlloc2(height, width);
	float a, b, c, d, t1, t2;
	a = 2; b = 0; c = 0; d = 2;
	t1 = 0; t2 = 0;
	//t1 = width, t2 = height;


	for (int y2 = 1; y2 < height; y2++)
		for (int x2 = 1; x2 < width; x2++)
		{
			int x = 1 / (a*d - b * c)*(d*(x2 - t1) - b * (y2 - t2));
			int y = 1 / (a*d - b * c)*((-1)*c*(x2 - t1) + a * (y2 - t2));

			if (x >= width || y >= height || x < 0 || y < 0) continue;
			else image_out[y2][x2] = BilinearInterpolation(image, width, height, x, y);
		}


	ImageShow((char*)"test", image, height, width);
	ImageShow((char*)"test", image_out, height, width);
}

//가운데 중심으로 돌리기 -> 숙제

void main_0320()
{

	int width, height, width_out, height_out;
	float x, y;
	int** image = ReadImage((char*)"pG.jpg", &width, &height);
	int** image_out = IntAlloc2(width, height);

	double theta = 45 * (3.14 / 180);


	for (int v = 0; v < height; v++)
		for (int u = 0; u < width; u++)
		{
			x = cos(theta)*(u - width / 2) + sin(theta)*(v - height / 2) + width / 2;
			y = sin(theta)*(u - width / 2) + cos(theta)*(v - height / 2) + height / 2;

			if (x >= width || y >= height || x < 0 || y < 0) continue;
			else image_out[v][u] = BilinearInterpolation(image, width, height, (int)x, (int)y);
		}


	ImageShow((char*)"original", image, width, height);
	ImageShow((char*)"output", image_out, width, height);



	IntFree2(image, width, height);
	IntFree2(image_out, width, height);
}
/*
x' = (a b)  x  + t1
y'	 (c d)  y	   t2
에서
(x'-w/2-t1) = (a b)  (x-w/2) ==>  (a b)-1(x'-w/2-t1) + (w/2) = (x) ==> 1(d -b)(x' - w/2 - t1) + w/2
(y'-h/2-t1) = (c d)  (y-h/2)      (c d)  (y'-h/2-t2)   (h/2)   (y)     D(-c a)(y' - h/2 - t1) + h/2
*/
void main_0320_2()
{
	int height, width;
	int** image = ReadImage((char*)"pG.jpg", &height, &width);
	int** image_out = IntAlloc2(height, width);
	float a, b, c, d, t1, t2;
	double theta = 45 * (3.14 / 180);
	a = cos(theta);
	b = (-1)*sin(theta);
	c = sin(theta);			//크기를 많이 줄이면 -> 반짝거리는 현상(에얼리어싱)이 발생한다.
	d = cos(theta);			//해결하려면 Lowpass Filtering이 필요하다.
	double det = a * d - b * c;
	t1 = 0; t2 = 0;
	//t1 = width, t2 = height;


	for (int y2 = 1; y2 < height; y2++)
		for (int x2 = 1; x2 < width; x2++)
		{
			int x = 1 / det * (d*(x2 - width / 2 - t1) - b * (y2 - height / 2 - t2)) + width / 2;
			int y = 1 / det * ((-1)*c*(x2 - width / 2 - t1) + a * (y2 - height / 2 - t2)) + height / 2;

			if (x >= width || y >= height || x < 0 || y < 0) continue;
			else image_out[y2][x2] = BilinearInterpolation(image, width, height, x, y);
		}


	ImageShow((char*)"test", image, height, width);
	ImageShow((char*)"test", image_out, height, width);
}

struct POS3D {
	float x, y, z;
};

POS3D SetPOS3D(float x, float y, float z)
{
	POS3D tmp;
	tmp.x = x; tmp.y = y; tmp.z = z;
	return (tmp);
}

POS3D PlusPOS3D(POS3D p, POS3D q)
{
	POS3D s;
	s.x = p.x + q.x;	s.y = p.y + q.y;	s.z = p.z + q.z;
	return s;
}
POS3D PlusPOS3DTriple(POS3D p, POS3D q, POS3D r)
{
	POS3D s;
	s.x = p.x + q.x + r.x;	s.y = p.y + q.y + r.y;	s.z = p.z + q.z + r.z;
	return s;
}
POS3D PlusPOS3DQuadra(POS3D p, POS3D q, POS3D r, POS3D s)
{
	POS3D t;
	t.x = p.x + q.x + r.x + s.x;	t.y = p.y + q.y + r.y + s.y;	t.z = p.z + q.z + r.z + s.z;
	return s;
}
POS3D MinusPOS3D(POS3D p, POS3D q)
{
	POS3D s;
	s.x = p.x - q.x;	s.y = p.y - q.y;	s.z = p.z - q.z;
	return s;
}

struct POS2D {
	float x, y;
};

POS2D SetPOS2D(float x, float y)
{
	POS2D tmp;
	tmp.x = x; tmp.y = y;;
	return (tmp);
}

POS2D PlusPOS2D(POS3D p, POS3D q)
{
	POS2D s;
	s.x = p.x + q.x;	s.y = p.y + q.y;
	return s;
}
POS2D MinusPOS2D(POS3D p, POS3D q)
{
	POS2D s;
	s.x = p.x - q.x;	s.y = p.y - q.y;
	return s;
}

POS3D Ax_3(float**A, POS3D p)//행렬 A와 구조체 x를 곱해주는 함수
{
	POS3D s;
	s.x = A[0][0] * p.x + A[0][1] * p.y + A[0][2] * p.z;
	s.y = A[1][0] * p.x + A[1][1] * p.y + A[1][2] * p.z;
	s.z = A[2][0] * p.x + A[2][1] * p.y + A[2][2] * p.z;
	return s;
}

#define SQ(x) ((x)*(x))
#define NP 1000 // Num of Points

void ImageShowFloat(char* winname, float** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}
void SetMatrixA(int width, int height, float** A)//2차원배열 A를 설정하는 함수
{
	int f = 300;
	POS2D p0;
	p0.x = width / 2; p0.y = height / 2;

	A[0][0] = f; A[0][1] = 0; A[0][2] = p0.x;
	A[1][0] = 0; A[1][1] = f; A[1][2] = p0.y;
	A[2][0] = 0; A[2][1] = 0; A[2][2] = 1;
}
void DrawDot(int**image, POS3D input, int width, int height)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if ((int)input.y < 0 || (int)input.x < 0 || (int)input.y >= height || (int)input.x >= width) continue;
			else image[(int)input.y][(int)input.x] = 255;
		}
	}
}
void DrawLinePQ(POS3D p, POS3D q, float**A, int**image, int height, int width)//두 점 p와 q를 잇는 선분을 그리는 함수
{
	float length = sqrt(SQ(p.x - q.x) + SQ(p.y - q.y) + SQ(p.z - q.z));
	float delta = length / NP;
	POS3D diff = MinusPOS3D(q, p); // q-p 차 벡터
								   /*
								   line[0].x = p.x + 0 * delta * diff.x;
								   line[0].y = p.y + 0 * delta * diff.y;
								   line[0].z = p.z + 0 * delta * diff.z;

								   line[1].x = p.x + 1 * delta * diff.x;
								   line[1].y = p.y + 1 * delta * diff.y;
								   line[1].z = p.z + 1 * delta * diff.z;
								   ... => for loop
								   */
								   /*
								   for (int i = 0; i < NP; i++)
								   {
								   line[i].x = p.x + i * delta * diff.x;
								   line[i].y = p.y + i * delta * diff.y;
								   line[i].z = p.z + i * delta * diff.z;
								   output[i] = Ax_3(A, line[i]);
								   //이를 이용하여 선분 그리기 -> 과제
								   DrawDot(image, line[i], width, height);
								   }
								   ImageShow((char*)"test", image, height, width);
								   */
								   //190402
								   //더 간단하게 하면
	for (int i = 0; i < NP; i++) {
		POS3D output, line;
		float t = (float)i / NP;
		line.x = p.x + t * diff.x;
		line.y = p.y + t * diff.y;
		line.z = p.z + t * diff.z;

		output = Ax_3(A, line);

		int x = (int)(output.x / output.z + 0.5);
		int y = (int)(output.y / output.z + 0.5);

		if (x >= width || x < 0 || y >= height || y < 0)continue;
		else
		{
			image[y][x] = 255;
			//printf("\n(x,y) = (%d, %d)", x, y);
		}
	}
	//ImageShow((char*)"output", image, height, width);
}
void DrawRectangle(POS3D p1, POS3D p2, POS3D p3, POS3D p4, float**A, int**image, int height, int width)//p1-p2, p2-p4, p3-p4, p3-p1을 연결하여 직사각형을 그려주는 함수
{
	DrawLinePQ(p1, p2, A, image, height, width);
	DrawLinePQ(p2, p4, A, image, height, width);
	DrawLinePQ(p3, p4, A, image, height, width);
	DrawLinePQ(p3, p1, A, image, height, width);
}
void main_0402()
{
	int height = 500; int width = 500;
	int** image = (int**)IntAlloc2(height, width);
	float** A = (float**)FloatAlloc2(3, 3);
	/*
	float qx, qy, qz;
	float px, py, pz;
	float sx, sy, sz;

	qx = 1.0; qy = 2.0; qz = 3.0;
	px = 2.0; py = 2.0; pz = 3.0;
	sx = qx + px;
	sy = qy + py;
	sz = qz + pz;
	printf("\n(%f, %f, %f) + (%f, %f, %f) = (%f, %f, %f)", qx, qy, qz, px, py, pz, sx, sy, sz);
	*/
	//위와 같이 하면 불편. 구조체 이용
	/*
	struct POS3D {
	float x, y, z;
	};

	POS3D p, q, s;
	p.x = 1.0; p.y = 2.0; p.z = 3.0;
	q.x = 2.0; q.y = 2.0; q.z = 3.0;
	s.x = q.x + p.x;
	s.y = q.y + p.y;
	s.z = q.z + p.z;
	printf("\n(%f, %f, %f) + (%f, %f, %f) = (%f, %f, %f)", q.x, q.y, q.z, p.x, p.y, p.z, s.x, s.y, s.z);
	*/
	//이 역시 크게 다르지 않다. 함수화
	SetMatrixA(width, height, A);
	POS3D p1 = SetPOS3D(100, 200, 600);
	POS3D q1 = SetPOS3D(100, -200, 500);
	DrawLinePQ(p1, q1, A, image, height, width);

	POS3D q2 = SetPOS3D(100, -100, 400);
	POS3D p2 = SetPOS3D(100, 300, 500);
	DrawLinePQ(p2, q2, A, image, height, width);

	POS3D p3 = SetPOS3D(100, 300, 600);
	POS3D q3 = SetPOS3D(100, -200, 500);
	DrawLinePQ(p3, q3, A, image, height, width);

	POS3D p4 = SetPOS3D(100, 100, 500);
	POS3D q4 = SetPOS3D(100, 200, 300);
	DrawLinePQ(p4, q4, A, image, height, width);

	ImageShow((char*)"output", image, height, width);
}

void main_0402_2()//이 방식은 기준점(p1)에서부터 x,y축으로 i만큼 벌어진 직사각형을 여러 번 그려서 채우는 방식으로, 원하는 바와는 조금 다르다.
{
	int height = 1000; int width = 1000;
	int** image = (int**)IntAlloc2(height, width);
	float** A = (float**)FloatAlloc2(3, 3);
	SetMatrixA(width, height, A);
	POS3D a, b, p1, p2, p3, p4;

	for (int i = 0; i < 100; i++) {
		a = SetPOS3D(i, 0, 0);
		b = SetPOS3D(0, i, 0);

		p1 = SetPOS3D(100, 200, 600);
		p2 = PlusPOS3D(p1, a);
		p3 = PlusPOS3D(p1, b);
		p4 = PlusPOS3DTriple(p1, a, b);

		DrawRectangle(p1, p2, p3, p4, A, image, height, width);
	}
	ImageShow((char*)"output", image, height, width);
}
void main_0402_3()//p1에서 a, b만큼 떨어진 점 x를 여러 번 그리는 방식 -> 부적절
{
	int height = 1000; int width = 1000;
	int** image = (int**)IntAlloc2(height, width);
	float** A = (float**)FloatAlloc2(3, 3);
	SetMatrixA(width, height, A);
	POS3D a, b, p;
	p = SetPOS3D(100, 200, 600);

	for (float i = 0; i < 200; i++)
	{
		for (float j = 0; j < 150; j++)
		{
			POS3D x, output;
			a = SetPOS3D(i, 0, 0);
			b = SetPOS3D(0, j, 0);
			x = PlusPOS3DTriple(p, a, b);
			output = Ax_3(A, x);
			DrawDot(image, output, width, height);
		}
	}
	ImageShow((char*)"output", image, height, width);
}
POS3D MultiPOS3D(float a, POS3D p)
{
	POS3D s;
	s.x = p.x * a; s.y = p.y * a; s.z = p.y * a;
	return s;
}
//p1,p2,p3좌표(직사각형이므로)를 주어주고, 그 내부의 점들을 모두 계산하는 방식, 함수화 할 것, 내부에 사진 넣을 것 ==> 렌더링(Rendering) : 3차원상의 영상을 2차원으로

float SizeofVector(POS3D p1, POS3D p2)
{
	float result = sqrt(SQ(p2.x - p1.x) + SQ(p2.y - p1.y) + SQ(p2.z - p1.z));
	return result;
}

void ResizeImage(int**image, int**image_out, int height, int width, int height_out, int width_out)
{
	for (int i = 0; i < height_out; i++)
	{
		for (int j = 0; j < width_out; j++)
		{
			int x = j * ((float)width / (float)width_out);
			int y = i * ((float)height / (float)height_out);
			if (j < 0 || j >= width_out || i < 0 || i >= height_out || x < 0 || x >= width || y < 0 || y >= height) continue;
			else image_out[i][j] = image[y][x];//BilinearInterpolation(image, width, height, x, y);
		}
	}
}

void Rendering2D(POS3D p1, POS3D p2, POS3D p3, int**image, int** image_out, int height, int width, int height_out, int width_out)
{
	float** A = (float**)FloatAlloc2(3, 3);
	SetMatrixA(width_out, height_out, A);
	POS3D v21, v31;

	v21 = MinusPOS3D(p2, p1);
	v31 = MinusPOS3D(p3, p1);
	//일단 image를 해당 영역의 크기에 맞게 변형시킨다. width => |p2 - p1|, height => |p3 - p1|
	int width_temp = width;
	int height_temp = height;
	int** image_temp = (int**)IntAlloc2(width_temp, height_temp);
	width = (int)SizeofVector(p1, p2);
	height = (int)SizeofVector(p1, p3);
	ResizeImage(image, image_temp, height_temp, width_temp, height, width);
	//test
	//	ImageShow((char*)"image_temp", image_temp, height, width);
	//	printf("\nwidth = %d, height = %d", width, height);

	for (float t2 = 0.0; t2 < 1.0; t2 += (float)1 / NP)
	{
		for (float t1 = 0.0; t1 < 1.0; t1 += (float)1 / NP)
		{
			POS3D q, output;
			q = PlusPOS3DTriple(p1, MultiPOS3D(t1, v21), MultiPOS3D(t2, v31));
			output = Ax_3(A, q);

			int x = (int)(output.x / output.z + 0.5);
			int y = (int)(output.y / output.z + 0.5);

			if (x >= width_out || x < 0 || y >= height_out || y < 0)continue;
			else
			{
				image_out[y][x] = image_temp[(int)(t2*height)][(int)(t1*height)];
			}
		}
	}
}

//위와 비슷한 방식으로 큐브 만들고, 면에다가 사진을 넣을 것(추가점수), 과제 제출할 때 소스랑 결과 캡쳐
void Rendering3D(POS3D p1, POS3D p2, POS3D p3, POS3D p4, int**image, int**image_out, int height, int width, int height_out, int width_out)
{
	int** image_temp1 = (int**)IntAlloc2(height_out, width_out);
	int** image_temp2 = (int**)IntAlloc2(height_out, width_out);
	int** image_temp3 = (int**)IntAlloc2(height_out, width_out);

	Rendering2D(p1, p2, p3, image, image_out, height, width, height_out, width_out);
	Rendering2D(p1, p2, p4, image, image_out, height, width, height_out, width_out);
	Rendering2D(p1, p3, p4, image, image_out, height, width, height_out, width_out);

	/*
	float** A = (float**)FloatAlloc2(3, 3);
	SetMatrixA(width_out, height_out, A);
	POS3D v21, v31, v41;

	v21 = MinusPOS3D(p2, p1);
	v31 = MinusPOS3D(p3, p1);
	v41 = MinusPOS3D(p4, p1);

	int width_temp = width;
	int height_temp = height;
	int** image_temp = (int**)IntAlloc2(width_temp, height_temp);
	width = (int)SizeofVector(p1, p2);
	height = (int)SizeofVector(p1, p3);
	ResizeImage(image, image_temp, height_temp, width_temp, height, width);

	for (float t1 = 0.0; t1 < 1.0; t1 += (float)1 / NP)
	{
	for (float t2 = 0.0; t2 < 1.0; t2 += (float)1 / NP)
	{
	for (float t3 = 0.0; t3 < 1.0; t3 += (float)1 / NP)
	{
	POS3D q, output;
	q = PlusPOS3DQuadra(p1, MultiPOS3D(t1, v21), MultiPOS3D(t2, v31), MultiPOS3D(t3, v41));
	output = Ax_3(A, q);

	int x = (int)(output.x / output.z + 0.5);
	int y = (int)(output.y / output.z + 0.5);

	if (x >= width || x < 0 || y >= height || y < 0)continue;
	else
	{
	image_out[y][x] = image_temp[(int)(t2*height)][(int)(t1*height)];
	}
	}

	}
	}
	*/
}

void main_0408()
{
	int height, width;
	int** image = ReadImage((char*)"cat.jpg", &height, &width);
	int height_out = 1000; int width_out = 1000;
	int** image_out_2D = (int**)IntAlloc2(height_out, width_out);
	int** image_out_3D = (int**)IntAlloc2(height_out, width_out);

	//2D////////////////////////
	POS3D p11, p12, p13;
	p11 = SetPOS3D(100, 200, 600);
	p12 = SetPOS3D(600, 200, 600);
	p13 = SetPOS3D(100, 600, 600);
	//3D////////////////////////
	POS3D p21, p22, p23, p24;
	p21 = SetPOS3D(100, 200, 600);
	p22 = SetPOS3D(600, 200, 600);
	p23 = SetPOS3D(100, 600, 600);
	p24 = SetPOS3D(300, 300, 800);

	Rendering2D(p11, p12, p13, image, image_out_2D, height, width, height_out, width_out);
	Rendering3D(p21, p22, p23, p24, image, image_out_3D, height, width, height_out, width_out);
	ImageShow((char*)"image", image, height, width);
	ImageShow((char*)"image_out_2D", image_out_2D, height_out, width_out);
	ImageShow((char*)"image_out_3D", image_out_3D, height_out, width_out);
}

void DownSize2(int**image, int height, int width, int**image_out)
{
	int height_out = height / 2, width_out = width / 2;
	for (int i = 0; i < height; i += 2)
	{
		for (int j = 0; j < width; j += 2)
		{
			image_out[(int)i / 2][(int)j / 2] = (image[i][j] + image[i][j + 1] + image[i + 1][j] + image[i + 1][j + 1]) / 4;
		}
	}
}

void DownSizeN(int**image, int height, int width, int n, int**image_out)
{
	int height_out = height / n, width_out = width / n;
	for (int i = 0; i < height; i += n)
	{
		for (int j = 0; j < width; j += n)
		{
			for (int k = 0; k < n; k++)
			{
				for (int l = 0; l < n; l++)
				{
					image_out[(int)i / n][(int)j / n] += image[i + k][j + l];
				}
			}
			image_out[(int)i / n][(int)j / n] /= n * n;


		}
	}
}

void main_0409()//190409 DownSampling
{
	int height, width, height_out, width_out;
	int** image = ReadImage((char*)"pG.jpg", &height, &width);
	int** image_out = (int**)IntAlloc2(height, width);
	int n = 5;

	width_out = width / n;
	height_out = height / n;

	DownSizeN(image, height, width, n, image_out);

	ImageShow((char*)"image", image, height, width);
	ImageShow((char*)"image_out", image_out, height_out, width_out);
}


int FindErr(int**image, int**block, int x, int y, int width, int height, int width_t, int height_t)
{
	int Err = 0;
	for (int i = 0; i < height_t; i++)
	{
		for (int j = 0; j < width_t; j++)
		{
			if (x + j >= width || y + i >= height)continue;
			else Err += abs(image[y + i][x + j] - block[i][j]);
		}
	}
	return Err;
}

POS2D TemplateMatching(int**image, int height, int width, int**block, int height_t, int width_t, int*err_min)
{
	int err;
	*err_min = INT_MAX;
	POS2D pos;
	for (int i = 0; i < height - height_t; i++)
		for (int j = 0; j < width - width_t; j++)
		{
			err = FindErr(image, block, j, i, width, height, width_t, height_t);
			if (err > *err_min);
			else
			{
				*err_min = err;
				pos = SetPOS2D(j, i);
			}
		}
	return pos;

}
void GeometricTransform(int**block, int**block_changed, int height_b, int width_b, int num)
//num 0:가만히 1:x축대칭 2:y축대칭 3:원점대칭 4:대각선대칭(/) 5:대각선대칭(반대) 6:90도 회전 7:180도 회전
{
	//	scanf_s("%d", &num);
	switch (num) {

	case 0:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[i][j];
			}
		}
		break;
	case 1:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[height_b - 1 - i][j];
			}
		}
		break;
	case 2:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[i][width_b - 1 - j];
			}
		}
		break;
	case 3:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				if (j > height_b || i > width_b) continue;
				else block_changed[i][j] = block[j][i];
			}
		}
		break;
	case 4:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[width_b - 1 - j][height_b - 1 - i];
			}
		}
		break;
	case 5:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[j][height_b - 1 - i];
			}
		}
		break;
	case 6:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[height_b - 1 - i][width_b - 1 - j];
			}
		}
		break;
	case 7:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[height_b - 1 - j][i];
			}
		}
		break;
	default:
		printf("\nWrong number");
		break;
	}
}
void CoverImageWithTemplate(int**image, int**block, int height, int width, int height_b, int width_b, POS2D pos)
{
	for (int i = 0; i < height_b; i++)
	{
		for (int j = 0; j < width_b; j++)
		{
			if ((int)pos.y + i >= height || (int)pos.x + j >= width || (int)pos.y + i < 0 || (int)pos.x + j < 0) continue;
			else image[(int)pos.y + i][(int)pos.x + j] = block[i][j];
		}
	}
}
void LightenImage(int**image, int height, int width, int num)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (image[i][j] > 255) image[i][j] = 255;
			else image[i][j] += num;
		}
	}
}

void main_0501()
{
	int height, width, height_t, width_t, height_a, width_a;
	int** image = ReadImage((char*)"Koala.bmp", &height, &width);
	int** block = ReadImage((char*)"template(flipping).bmp", &height_t, &width_t);
	int** GTTestBlock = ReadImage((char*)"LENA256.bmp", &height_a, &width_a);
	int err[8];
	int** block_changed[8];
	POS2D pos[8];
	//num 0:가만히 1:x축대칭 2:y축대칭 3:원점대칭 4:대각선대칭(/) 5:대각선대칭(반대) 6:90도 회전 7:180도 회전
	for (int i = 0; i < 8; i++)
	{
		block_changed[i] = (int**)IntAlloc2(height_t, width_t);
		GeometricTransform(block, block_changed[i], height_t, width_t, i + 1);
		pos[i] = TemplateMatching(image, height, width, block_changed[i], height_t, width_t, &err[i]);

		if (err[i] == 0)
		{
			LightenImage(block_changed[i], height_t, width_t, 30);//잘 보이게 하기 위해서
			CoverImageWithTemplate(image, block_changed[i], height, width, height_t, width_t, pos[i]);
			printf("\nAnswer is : %d", i);
			printf("\nAnd Err is %d, pos = (%f, %f)", err[i], pos[i].x, pos[i].y);
		}
		//		ImageShow((char*)"image", image, height, width);
	}

	ImageShow((char*)"image", image, height, width);
	//	for (int i = 0; i < 8; i++) ImageShow((char*)"image", block_changed[i], height, width);
}

int GetBlockAvg(int** image, int height, int width, int y, int x, int N)
{
	int avg = 0;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (y + i > height || x + j > width) continue;
			else avg += image[y + i][x + j];
		}
	}
	avg /= N*N;
	return avg;
}

void main()//190507
{
	int height, width, N;
	int** image = ReadImage((char*)"LENA256.bmp", &height, &width);
	int** image_out = IntAlloc2(32, 32);
	int** block_out = IntAlloc2(16, 16);

	N = 8;
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			image_out[i][j] = GetBlockAvg(image, height, width, N*i, N*j, N);
		}
	}
	
	ImageShow((char*)"image_out", image_out, 32, 32);

	DownSize2(image_out, 32, 32, block_out);

	ImageShow((char*)"block_out", block_out, 16, 16);
}

