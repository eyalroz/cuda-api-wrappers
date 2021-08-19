/**
 * @file devices.hpp
 *
 * @brief Code regarding the entirety of CUDA devices available on a system.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DEVICES_HPP_
#define CUDA_API_WRAPPERS_DEVICES_HPP_

#include <cuda/api/device.hpp>

namespace cuda {

namespace detail_ {

// Note that while nothing constrains you from instantiating
// this class many times, all instances are the same (as CUDA
// devices aren't hot-pluggable).
class all_devices {
public:
	using value_type = cuda::device_t;
	using pointer = void; // No pointers, since we don't have any elements in actual memory
	using const_pointer = void; // ditto
	using reference = value_type; // device_t is already a reference type; and there is no instance-of-device_t here to reference
	using const_reference = const value_type; // ditto
	using size_type = decltype(device::count());
	using difference_type = ::std::ptrdiff_t;

	class index_based_iterator {
	public:
		using container = all_devices;
		using difference_type = container::difference_type;
		using value_type = container::value_type;
		using pointer = container::pointer;
		using reference = container::reference;
		using iterator_category = ::std::random_access_iterator_tag;

		// Note: the sentinel iterator value has an index equal to num_indices

		// something about the traits

		index_based_iterator(size_type num_devices, size_type index)
			: num_devices_(num_devices), index_(index)
		{
			if (index_ > num_devices_) { throw ::std::logic_error("Out of range"); }
		}

		index_based_iterator(const index_based_iterator& it)
			: index_based_iterator(it.num_devices_, it.index_) { }

		// Forward iterator requirements

		reference operator*() const { return device::get(index_); }

		index_based_iterator&  operator++()
		{
			if (index_ == num_devices_) { throw ::std::logic_error("Out of range"); }
			++index_;
			return *this;
		}

		index_based_iterator operator++(int)
		{
			if (index_== num_devices_) { throw ::std::logic_error("Out of range"); }
			return index_based_iterator(num_devices_, index_++);
		}

		// Bidirectional iterator requirements
		index_based_iterator& operator--()
		{
			if (index_ == 0) { throw ::std::logic_error("Out of range"); }
			--index_;
			return *this;
		}

		index_based_iterator operator--(int)
		{
			if (index_ == 0) { throw ::std::logic_error("Out of range"); }
			return index_based_iterator(num_devices_, index_--);
		}

		// Random access iterator requirements
		reference operator[](difference_type n) const
		{
			return device::get(index_ + n);
		}

		index_based_iterator& operator+=(difference_type n)
		{
			if (index_ + n > num_devices_) { throw ::std::logic_error("Out of range"); }
			index_ += n;
			return *this;
		}

		index_based_iterator operator+(difference_type n) const
		{
			if (n + index_ > num_devices_) {
				throw ::std::logic_error("Out of range");
			}
			return index_based_iterator(num_devices_, index_ + n);
		}

		index_based_iterator& operator-=(difference_type n)
		{
			if (n > index_) {
				throw ::std::logic_error("Out of range");
			}
			index_ -= n;
			return *this;
		}

		index_based_iterator operator-(difference_type n) const
		{
			if (n > index_) {
				throw ::std::logic_error("Out of range");
			}
			return index_based_iterator(num_devices_, index_ - n);
		}

		size_type index() const { return index_; }
		size_type num_devices() const { return num_devices_; }

	protected:
		size_type num_devices_;
		size_type index_;
	}; // class index_based_iterator

	using iterator = index_based_iterator;
	using const_iterator = index_based_iterator;
	using reverse_iterator = ::std::reverse_iterator<iterator>;
	using const_reverse_iterator = ::std::reverse_iterator<const_iterator>;


	all_devices() : num_devices_(device::count()) { }
	~all_devices() = default;
	all_devices(const all_devices&) = default;
	all_devices(all_devices&&) = default;
	all_devices& operator=(const all_devices&) { return *this; };
	all_devices& operator=(all_devices&&) { return *this; };

	// void fill(const value_type& u);
	void swap(all_devices&) noexcept { } // all instances are basically the same

	// Iterators

	iterator begin() noexcept { return iterator(num_devices_, 0); }
	const_iterator begin() const noexcept { return const_iterator(num_devices_, 0); }
	iterator end() noexcept { return iterator(num_devices_, num_devices_); }
	const_iterator end() const noexcept { return const_iterator(num_devices_, num_devices_); }
	reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
	const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
	reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
	const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
	const_iterator cbegin() const noexcept	{ return const_iterator(num_devices_, 0); }
	const_iterator cend() const noexcept { return const_iterator(num_devices_, num_devices_); }
	const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
	const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

	// Capacity

	size_type size() const noexcept { return num_devices_; }
	size_type max_size() const noexcept { return num_devices_; }
	bool empty() const noexcept { return size() == 0; }

	// Element access

	// reference at(size_type n);
	const_reference at(size_type n) const
	{
		// Note; this getter will throw if the device index is out-of-range
		return device::get(n);
	}

	//	reference operator[](size_type i);
		const_reference operator[](size_type n) const { return at(n); }


	// reference front();
	const_reference front() const { return *begin(); }

	const_reference back() const noexcept { return num_devices_ ? *(end() - 1) : *end(); }
	// reference back() noexcept;

protected:
	size_type num_devices_;
};

inline bool operator== (
	const all_devices::index_based_iterator& lhs,
	const all_devices::index_based_iterator& rhs)
{
#ifndef NDEBUG
	return lhs.num_devices() == rhs.num_devices() and lhs.index() == rhs.index();
#else
	return lhs.index() == rhs.index();
#endif
}

inline bool operator!= (
	const all_devices::index_based_iterator& lhs,
	const all_devices::index_based_iterator& rhs)
{
	return not (lhs == rhs);
}

} // namespace detail_

inline detail_::all_devices devices()
{
	return detail_::all_devices();
}

} // namespace cuda

#endif // CUDA_API_WRAPPERS_DEVICES_HPP_
