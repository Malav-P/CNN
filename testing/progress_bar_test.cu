//
// Created by malav on 10/1/2022.
//

#include "../classes/helpers/progress_bar.hxx"
#include <thread>

int main() {

    ProgressBar bar;
    bar.set_bar_width(50);
    bar.fill_bar_progress_with("■");
    bar.fill_bar_remainder_with(" ");

    for (size_t i = 1; i <= 100; ++i) {
        bar.update(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << "\n";

}