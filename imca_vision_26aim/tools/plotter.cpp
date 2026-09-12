#include "plotter.hpp"

#include <arpa/inet.h>   // htons, inet_addr
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <sys/socket.h>  // socket, sendto
#include <unistd.h>      // close
#include <vector>

namespace tools
{
namespace
{
constexpr unsigned char kJustFloatTail[4] = {0x00, 0x00, 0x80, 0x7F};

void append_channel(const nlohmann::json & value, std::vector<float> & channels)
{
  if (value.is_number_float()) {
    channels.push_back(static_cast<float>(value.get<double>()));
  } else if (value.is_number_integer()) {
    channels.push_back(static_cast<float>(value.get<long long>()));
  } else if (value.is_number_unsigned()) {
    channels.push_back(static_cast<float>(value.get<unsigned long long>()));
  } else if (value.is_boolean()) {
    channels.push_back(value.get<bool>() ? 1.0F : 0.0F);
  }
}

void append_float_le(float value, std::vector<unsigned char> & frame)
{
  std::array<unsigned char, sizeof(float)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(float));

  if constexpr (std::endian::native == std::endian::little) {
    frame.insert(frame.end(), bytes.begin(), bytes.end());
  } else {
    frame.insert(frame.end(), bytes.rbegin(), bytes.rend());
  }
}
}  // namespace

Plotter::Plotter(std::string host, uint16_t port)
{
  socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);

  destination_.sin_family = AF_INET;
  destination_.sin_port = ::htons(port);
  destination_.sin_addr.s_addr = ::inet_addr(host.c_str());
}

Plotter::~Plotter() { ::close(socket_); }

void Plotter::plot(const nlohmann::json & json)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto data = json.dump();
  ::sendto(
    socket_, data.c_str(), data.length(), 0, reinterpret_cast<sockaddr *>(&destination_),
    sizeof(destination_));
}

void Plotter::vofa(const nlohmann::json & json)
{
  std::vector<float> channels;

  if (json.is_object()) {
    channels.reserve(json.size());
    for (const auto & item : json.items()) {
      append_channel(item.value(), channels);
    }
  } else if (json.is_array()) {
    channels.reserve(json.size());
    for (const auto & value : json) {
      append_channel(value, channels);
    }
  } else {
    append_channel(json, channels);
  }

  std::vector<unsigned char> frame;
  frame.reserve(channels.size() * sizeof(float) + sizeof(kJustFloatTail));
  for (float value : channels) {
    append_float_le(value, frame);
  }
  frame.insert(frame.end(), kJustFloatTail, kJustFloatTail + sizeof(kJustFloatTail));

  std::lock_guard<std::mutex> lock(mutex_);
  ::sendto(
    socket_, reinterpret_cast<const char *>(frame.data()), frame.size(), 0,
    reinterpret_cast<sockaddr *>(&destination_),
    sizeof(destination_));
}

}  // namespace tools
