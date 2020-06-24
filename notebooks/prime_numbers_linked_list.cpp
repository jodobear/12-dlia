#include <iostream>

using namespace std;

bool isPrime(int n) {
    if (n < 2) {
        return false;
    }
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}

struct Node {
    Node* nextNode;
    int value;
};

int main() {
    int n;
    cin >> n;

    if (n < 2) {
        return 0;
    }
    Node first;
    first.value = 2;
    first.nextNode = nullptr;

    Node* pointer = &first;
    for (int i = 3; i <= n; ++i) {
        if (isPrime(i)) {
            pointer->nextNode = new Node;
            pointer = pointer->nextNode;
            pointer->value = i;
            pointer->nextNode = nullptr;
        }
    }

    for (Node* ptr = &first; ptr; ptr = ptr->nextNode) {
        cout << ptr->value << ' ';
    }
}
