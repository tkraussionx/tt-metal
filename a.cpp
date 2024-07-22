#include <tuple>
#include <iostream>


template <typename object_t, typename... Ts>
constexpr auto visit_object_of_type(const std::tuple<Ts...>& value) {
    // constexpr auto num_attributes = std::tuple_size_v<decltype(std::decay_t<T>::attribute_names)>;
    // visit_object_of_type<object_t>(callback, object.attribute_values());

    std::cout << "AAAAAAAAAA 1 \n";
}

template <typename object_t, typename... Ts>
constexpr auto visit_object_of_type(std::tuple<Ts...>& value) {
    // constexpr auto num_attributes = std::tuple_size_v<decltype(std::decay_t<T>::attribute_names)>;
    // visit_object_of_type<object_t>(callback, object.attribute_values());

    std::cout << "AAAAAAAAAA 3 \n";
}

template <typename object_t, typename T>
constexpr auto visit_object_of_type(T&& object) {
    // constexpr auto num_attributes = std::tuple_size_v<decltype(std::decay_t<T>::attribute_names)>;
    // visit_object_of_type<object_t>(callback, object.attribute_values());

    std::cout << "AAAAAAAAAA 2 \n";
}

template <typename Tuple, typename T>
constexpr bool is_homogenous_tuple() {
    return []<std::size_t... Ns>(std::index_sequence<Ns...>) {
        return (std::is_same_v<T, std::tuple_element_t<Ns, Tuple>> && ...);
    }(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

int main()
{
    std::tuple<int, int> value; // 2번으로 가버림 안됨
    // const std::tuple<int, int> value; // 이렇게 해야 tuple 버전으로 호출됨
    visit_object_of_type<int>(value);

    // is_homogenous_tuple<execute_on_worker_thread_return_t, Tensor>()
    std::cout << "RRRRRRRRRRR " << is_homogenous_tuple<std::tuple<int, int>, int>() << "\n";

    return 0;
}
