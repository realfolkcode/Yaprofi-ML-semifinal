#include <iostream>
#include <string>
#include <functional>
#include <set>
#include <bitset>

#define HASH_SIZE 1000003

using namespace std;

int main()
{
	int n, i;
	string s;
	bitset<HASH_SIZE> bs;
	cin >> n;
	getline(cin, s);
	for (i = 0; i < n; ++i)
	{
		getline(cin, s);
		bs.set(hash<string>{}(s) % HASH_SIZE);
	}
	cout << bs.count();
	return 0;
}