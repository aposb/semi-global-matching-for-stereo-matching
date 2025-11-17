// Semi-Global Matching for Stereo Matching
//

#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef vector<int> Table1;
typedef vector<Table1> Table2;
typedef vector<Table2> Table3;

Mat leftImage, rightImage, disparityMap, disparityImage;
Table3 matchingCost, direction1, direction2, direction3, direction4;
Table3 direction5, direction6, direction7, direction8;
int width, height, levels, P1, P2;

void computeLinearPathCost(Table1 &C, Table1 &L, Table1 &out);
int computeMatchingCost(int x, int y, int d);
int findBestAssignment(int x, int y);
int computeAggregatedCost(int x, int y, int d);

int main()
{
	levels = 16;
	P1 = 5;
	P2 = 10;

	// Start timer
	auto start = chrono::steady_clock::now();

	// Read stereo image
	leftImage = imread("left.png", IMREAD_GRAYSCALE);
	rightImage = imread("right.png", IMREAD_GRAYSCALE);

	// Use gaussian filter
	GaussianBlur(leftImage, leftImage, Size(5, 5), 0.68);
	GaussianBlur(rightImage, rightImage, Size(5, 5), 0.68);

	// Get image size
	width = leftImage.cols;
	height = leftImage.rows;

	// Cache matching cost
	matchingCost = Table3(height, Table2(width, Table1(levels)));
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			for (int i = 0; i < levels; i++)
				matchingCost[y][x][i] = computeMatchingCost(x, y, i);

	// Initialize disparity map
	disparityMap = Mat::zeros(height, width, CV_8U);

	// Initialize tables for the 8 directions
	direction1 = Table3(height, Table2(width, Table1(levels, 0)));;
	direction2 = Table3(height, Table2(width, Table1(levels, 0)));;
	direction3 = Table3(height, Table2(width, Table1(levels, 0)));;
	direction4 = Table3(height, Table2(width, Table1(levels, 0)));;
	direction5 = Table3(height, Table2(width, Table1(levels, 0)));;
	direction6 = Table3(height, Table2(width, Table1(levels, 0)));;
	direction7 = Table3(height, Table2(width, Table1(levels, 0)));;
	direction8 = Table3(height, Table2(width, Table1(levels, 0)));;

	// Forward pass
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			if (x > 0)
				computeLinearPathCost(matchingCost[y][x], direction1[y][x - 1], direction1[y][x]);
			if (y > 0 && x > 0)
				computeLinearPathCost(matchingCost[y][x], direction2[y - 1][x - 1], direction2[y][x]);
			if (y > 0)
				computeLinearPathCost(matchingCost[y][x], direction3[y - 1][x], direction3[y][x]);
			if (y > 0 && x < width - 1)
				computeLinearPathCost(matchingCost[y][x], direction4[y - 1][x + 1], direction4[y][x]);
		}

	// Backward pass
	for (int y = height - 1; y >= 0; y--)
		for (int x = width - 1; x >= 0; x--)
		{
			if (x < width - 1)
				computeLinearPathCost(matchingCost[y][x], direction5[y][x + 1], direction5[y][x]);
			if (y < height - 1 && x < width - 1)
				computeLinearPathCost(matchingCost[y][x], direction6[y + 1][x + 1], direction6[y][x]);
			if (y < height - 1)
				computeLinearPathCost(matchingCost[y][x], direction7[y + 1][x], direction7[y][x]);
			if (y < height - 1 && x > 0)
				computeLinearPathCost(matchingCost[y][x], direction8[y + 1][x - 1], direction8[y][x]);
		}

	// Update disparity map
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			int label = findBestAssignment(x, y);
			disparityMap.at<uchar>(y, x) = label;
		}

	// Update disparity image
	int scaleFactor = 256 / levels;
	disparityMap.convertTo(disparityImage, CV_8U, scaleFactor);

	// Show disparity image
	namedWindow("Disparity Image", WINDOW_NORMAL);
	imshow("Disparity Image", disparityImage);
	waitKey(1);

	// Save disparity image
	imwrite("disparity.png", disparityImage);

	// Stop timer
	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << "\nRunning Time: " << chrono::duration<double, milli>(diff).count() << " ms" << endl;

	waitKey(0);

	return 0;
}

// Computes the linear path cost (L) for a given direction
void computeLinearPathCost(Table1 &C, Table1 &L, Table1 &out)
{
	int minL = INT_MAX;
	for (int i = 0; i < levels; i++)
		if (L[i] < minL)
			minL = L[i];

	for (int i = 0; i < levels; i++)
	{
		int cost1 = (i > 0) ? L[i - 1] + P1 : INT_MAX;
		int cost2 = L[i];
		int cost3 = (i < levels - 1) ? L[i + 1] + P1 : INT_MAX;
		int cost4 = minL + P2;

		out[i] = min({ cost1, cost2, cost3, cost4 }) - minL + C[i];
	}
}

// Computes the matching cost between two imege pixels
int computeMatchingCost(int x, int y, int d)
{
	int leftPixel = leftImage.at<uchar>(y, x);
	int rightPixel = (x >= d) ? rightImage.at<uchar>(y, x - d) : 0;
	int cost = abs(leftPixel - rightPixel);

	return cost;
}

// Computes the aggregated cost (S) from all directions
int computeAggregatedCost(int x, int y, int d)
{
	int cost = direction1[y][x][d] + direction2[y][x][d] + direction3[y][x][d] + direction4[y][x][d]
		+ direction5[y][x][d] + direction6[y][x][d] + direction7[y][x][d] + direction8[y][x][d];

	// An improvement to the algorithm for better results
	if (y > 0 && y < height - 1 && x > 0 && x < width - 1)
		cost -= matchingCost[y][x][d] * 8;

	return cost;
}

// Finds the disparity with the minimum cost
int findBestAssignment(int x, int y)
{
	int d, min = INT_MAX;
	for (int i = 0; i < levels; i++)
	{
		int cost = computeAggregatedCost(x, y, i);
		if (cost < min)
		{
			d = i;
			min = cost;
		}
	}

	return d;
}