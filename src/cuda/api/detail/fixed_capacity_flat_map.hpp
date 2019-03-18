#pragma once
#include <sstream>
#include <array>
#include <algorithm>


// TODO: Avoid multimap functionality by searching before inserting.
// Currently, getters always return the first matching entry.
template <class Key, class Value, size_t capacity, class Compare = std::less<Key> >
class fixed_capacity_flat_map
{
public:
	using key_type = Key;
	using value_type = Value;
	using kv_pair_type = std::pair<key_type, value_type>;
	using container_type = std::array<kv_pair_type, capacity>;

	// Iterators
	using iterator = typename container_type::iterator;
	using const_iterator = typename container_type::const_iterator;
	using reverse_iterator = typename container_type::reverse_iterator;
	using const_reverse_iterator = typename container_type::const_reverse_iterator;

	fixed_capacity_flat_map(const std::initializer_list<kv_pair_type>& values)
	{
		for (auto& value : values)
		{
			insert(value);
		}
	}

	fixed_capacity_flat_map() { }
	// TODO: Support copy-construction from different-capacity maps
	fixed_capacity_flat_map(const fixed_capacity_flat_map& other) { 
		size_ = other.size();
		std::copy_n(this, other.size(), other); 
	}

	iterator insert(const kv_pair_type& val)
	{
		auto position = std::upper_bound(begin(), end(), val, comparator);
		insert_at(position, val);
		return position;
	}

	value_type& at(const key_type& key)
	{
		auto elem = find(key);
		if (elem == end())
		{
			throw_out_of_range_error(key, __PRETTY_FUNCTION__);
		}
		return elem->second;
	}

	iterator find(const key_type& key)
	{
		// TODO: Avoid copying here - use references
		kv_pair_type pair{key, value_type()};
		auto range = std::equal_range(begin(), end(), pair, comparator);
		if (range.first == range.second)
		{
			return end();
		}
		return range.first;
	}

	iterator erase(const_iterator position)
	{
		if (position == end() or size_ == 0)
		{
			throw_range_error(*position, __PRETTY_FUNCTION__);
		}
		auto non_const_iter = const_cast<iterator>(position);
		// TODO: Is std::copy the right thing to do here?
		std::copy(non_const_iter + 1, end(), non_const_iter);
		--size_;
		return non_const_iter;
	}

	iterator erase(const key_type& key) { return erase(find(key)); }

	value_type& operator[](const key_type& key)
	{
		kv_pair_type pair{key, value_type()};
		auto range = std::equal_range(begin(), end(), pair, comparator);
		if (range.first == range.second)
		{
			// value was not found, inserting it in the right location
			insert_at(range.first, pair);
		}
		return range.first->second;
	}

	// std::map compatibility
	template <class... Params>
	iterator erase(const_iterator position)   { return erase(position); }
	iterator erase(const key_type& key)       { return erase(key);      }
	iterator insert(const kv_pair_type& val)  { return insert(val);     }
	iterator find(const key_type& key)        { return find(key);       }

	void clear() { size_ = 0; }

	iterator begin()                 noexcept { return iterator(&sorted_data_[0]);                  }
	iterator end()                   noexcept { return iterator(&sorted_data_[size_]);              }
	reverse_iterator rbegin()        noexcept { return reverse_iterator(&sorted_data_[size_]);      }
	reverse_iterator rend()          noexcept { return reverse_iterator(&sorted_data_[0]);          }

	const_iterator cbegin()          const noexcept { return const_iterator(&sorted_data_[0]);             }
	const_iterator cend()            const noexcept { return const_iterator(&sorted_data_[size_]);         }
	const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(&sorted_data_[size_]); }
	const_reverse_iterator crend()   const noexcept { return const_reverse_iterator(&sorted_data_[0]);     }

	const_iterator begin()           const noexcept { return const_iterator (&sorted_data_[0]);             }
	const_iterator end()             const noexcept { return const_iterator(&sorted_data_[size_]);          }
	const_reverse_iterator rbegin()  const noexcept { return const_reverse_iterator(&sorted_data_[size_]);  }
	const_reverse_iterator rend()    const noexcept { return const_reverse_iterator(&sorted_data_[0]);      }

	size_t size()     const { return size_; }
	size_t max_size() const { return capacity; }
	bool   empty()    const { return size() == 0; }

private:

	void insert_at(const iterator& position, const kv_pair_type& val)
	{
		if (size() == capacity)
			throw_range_error(val, __PRETTY_FUNCTION__);
		std::copy_backward(position, end(), end() + 1);
		*position = val;
		++size_;
	}

	void throw_range_error(const kv_pair_type& val, const char* throwing_function)
	{
		std::stringstream error_message;
		error_message << throwing_function << " : Out of range! key = " << val.first << " value = " << val.second;
		throw std::range_error(error_message.str().c_str());
	}

	void throw_out_of_range_error(const key_type& key, const char* throwing_function)
	{
		std::stringstream error_message;
		error_message << throwing_function << " : Could not find object in map! key = " << key;
		throw std::out_of_range(error_message.str().c_str());
	}

	// TODO: Is this necessary?
	static bool comparator(const kv_pair_type& first, const kv_pair_type& second)
	{
		static Compare less;
		return less(first.first, second.first);
	}

	std::array<kv_pair_type, capacity> sorted_data_;
	size_t size_ = 0;
};