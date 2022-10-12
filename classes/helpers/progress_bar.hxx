//
// Created by malav on 10/1/2022.
//

#ifndef CNN_PROGRESS_BAR_HXX
#define CNN_PROGRESS_BAR_HXX

#include <iostream>

class ProgressBar {
public:
    void set_progress(float value) {
        progress_ = value;
    }
    void set_bar_width(size_t width) {
        bar_width_ = width;
    }

    void fill_bar_progress_with(const std::string& chars) {
        fill_ = chars;
    }

    void fill_bar_remainder_with(const std::string& chars) {
        remainder_ = chars;
    }

    void set_status_text(const std::string& status) {
        status_text_ = status;
    }

    void update(float value, std::ostream &os = std::cout) {
        set_progress(value);
        write_progress(os);
    }

    void write_progress(std::ostream &os = std::cout) {

        // No need to write once progress is 100%
        if (progress_ > 100.0f) return;

        // Move cursor to the first position on the same line and flush
        os << "\r" << std::flush;

        // Start bar
        os << "[";

        const auto completed = static_cast<size_t>(progress_ * static_cast<float>(bar_width_) / 100.0);
        for (size_t i = 0; i < bar_width_; ++i) {
            if (i <= completed)
                os << fill_;
            else
                os << remainder_;
        }

        // End bar
        os << "]";

        // Write progress percentage
        os << " " << std::min(static_cast<size_t>(progress_), size_t(100)) << "%";

        // Write status text
        os << " " << status_text_;
    }

private:

    float progress_{0.0f};
    size_t bar_width_ {60};
    std::string fill_{"#"}, remainder_{" "}, status_text_{""};
};

#endif //CNN_PROGRESS_BAR_HXX