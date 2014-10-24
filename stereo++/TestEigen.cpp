

#include <vector>
#include <stack>
#include <set>
#include <string>

#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseLU"
#include "Eigen/SparseQR"
#include "Eigen/Core"




void TestEigen()
{
	extern float ARAP_SIGMA;
	float G = 1;
	float maxDispSquare = 159 * 159;
	float val = ARAP_SIGMA * G * maxDispSquare;
	printf("%f\n", val);

	Eigen::SparseMatrix<float> A(100, 100);
	std::vector<Eigen::Triplet<float>> triplets(100);
	for (int i = 0; i < 100; i++) {
		triplets[i] = Eigen::Triplet<float>(i, i, 3 * i);
	}

	for (int i = 0; i < 10; i++) {
		//int idx = rand() % 100;
		int idx = 2 * i;
		//triplets[idx] = Eigen::Triplet<float>(idx, idx, 0);
		
	}
	A.setFromTriplets(triplets.begin(), triplets.end());
	A.makeCompressed();

	Eigen::MatrixXf b(100, 1);
	for (int i = 0; i < 100; i++) {
		b.coeffRef(i, 0) = 2 * i;
	}

	//Eigen::SparseLU<
	Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<Eigen::SparseMatrix<float>::Index>> qr(A);
	Eigen::MatrixXf X1 = qr.solve(b);

	/*Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> chol(A);
	Eigen::MatrixXf X2 = chol.solve(b);*/


	Eigen::SparseLU<Eigen::SparseMatrix<float>> lu(A);
	/*lu.analyzePattern(A);
	lu.factorize(A);*/
	Eigen::MatrixXf  X2 = lu.solve(b);
	printf("isnan = %d\n", isnan(X2.coeffRef(0, 0)));

	for (int i = 0; i < 100; i++) {
		printf("%10f   %10f   %10f\n", X1.coeffRef(i, 0), X2.coeffRef(i, 0), b.coeffRef(1, 0));
	}



}